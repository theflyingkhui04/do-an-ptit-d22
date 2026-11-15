"""Inspect checkpoint metadata without importing model code or downloading weights."""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

CACHE = Path.home() / '.flexsds' / 'cache' / 'inspection'
MAX_JSON = 32 * 1024 * 1024
MAX_HEADER = 8 * 1024 * 1024


class InspectionError(ValueError):
    pass


class ModelFiles:
    """A local directory or a Hub snapshot resolved to an immutable commit.

    Reads JSON, tensor headers and small vocabulary mappings. PyTorch ZIP
    checkpoints use a restricted metadata decoder; model Python and tensor
    storage are never loaded or executed.
    """
    def __init__(self, model):
        self.model = model
        path = Path(model).expanduser()
        self.local = (path.parent if path.is_file() else path) if path.exists() else None
        self.info = {}
        self.revision = 'local'
        if self.local is None:
            self.model = model.removeprefix('https://huggingface.co/').strip('/')
            if not re.fullmatch(r'[\w.-]+/[\w.-]+', self.model):
                raise InspectionError('Expected a local model directory or owner/model Hugging Face ID')
            key = hashlib.sha256(self.model.encode()).hexdigest()
            info_cache = CACHE / key / 'info.json'
            # Refresh mutable main; all artifact caches below are commit-specific.
            try:
                if time.time() - info_cache.stat().st_mtime < 300:
                    self.info = json.loads(info_cache.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
            if not self.info:
                try:
                    raw = self._http(f'https://huggingface.co/api/models/{self.model}?blobs=true', MAX_JSON)
                    self.info = json.loads(raw)
                    self._save(info_cache, raw)
                except (OSError, ValueError) as exc:
                    raise InspectionError(f'Cannot resolve model metadata for {model}: {exc}') from exc
            self.revision = self.info.get('sha', '')
            if not re.fullmatch(r'[0-9a-f]{40}', self.revision):
                raise InspectionError('Hub response does not identify an immutable model revision')
            self.cache = CACHE / key / self.revision

    @staticmethod
    def _save(path, raw):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except OSError:
            pass

    @staticmethod
    def _http(url, limit, byte_range=None):
        headers = {'User-Agent': 'FlexSDS-Inspection/0.2', 'Accept-Encoding': 'identity'}
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'
        if byte_range:
            start, end = byte_range
            headers['Range'] = f'bytes={start}-{end}'
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as response:
            if byte_range:
                expected = f'bytes {start}-{end}/'
                if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                    raise InspectionError('Server did not honor bounded checkpoint range request')
            raw = response.read(limit + 1)
        if len(raw) > limit:
            raise InspectionError('Metadata exceeds inspection size limit')
        if byte_range and len(raw) != end - start + 1:
            raise InspectionError('Truncated checkpoint range')
        return raw

    def files(self):
        if self.local is not None:
            return [str(p.relative_to(self.local)).replace(os.sep, '/') for p in self.local.rglob('*') if p.is_file()]
        return [s['rfilename'] for s in self.info.get('siblings', [])]

    def read(self, filename, limit=MAX_JSON, byte_range=None):
        if filename not in self.files():
            raise InspectionError(f'{filename} is missing in {self.model}')
        if self.local is not None:
            with (self.local / filename).open('rb') as stream:
                if byte_range:
                    stream.seek(byte_range[0])
                raw = stream.read(limit if byte_range else limit + 1)
            if len(raw) > limit:
                raise InspectionError('Local metadata exceeds inspection size limit')
            return raw
        suffix = '' if byte_range is None else f'.range-{byte_range[0]}-{byte_range[1]}'
        cached = self.cache / (filename + suffix)
        try:
            raw = cached.read_bytes()
            if len(raw) <= limit:
                return raw
        except OSError:
            pass
        url = f'https://huggingface.co/{self.model}/resolve/{self.revision}/{quote(filename)}'
        raw = self._http(url, limit, byte_range)
        self._save(cached, raw)
        return raw

    def json(self, filename):
        data = json.loads(self.read(filename))
        if not isinstance(data, dict):
            raise InspectionError(f'{filename} must contain a JSON object')
        return data

    def config(self):
        config = self.json('config.json')
        # Speculators stores transformer dimensions in a nested layer config.
        layer = config.get('transformer_layer_config')
        if isinstance(layer, dict):
            config = dict(layer, **config)
        return config

    def base_models(self, config):
        card = self.info.get('cardData') or {}
        verifier = (config.get('speculators_config') or {}).get('verifier') or {}
        value = (config.get('target_model_name_or_path') or config.get('base_model_name_or_path')
                 or verifier.get('name_or_path') or card.get('base_model'))
        if isinstance(value, str):
            return {value}
        if isinstance(value, list):
            return {v for v in value if isinstance(v, str)}
        return set()

    def file_size(self, filename):
        if self.local is not None:
            return (self.local / filename).stat().st_size
        return next((s.get('size', 0) for s in self.info.get('siblings', []) if s['rfilename'] == filename), 0)

    def weight_files(self):
        files = self.files()
        safe = [f for f in files if f.endswith('.safetensors') and '/' not in f]
        if safe:
            return safe
        return [f for f in files if re.fullmatch(r'pytorch_model(?:-\d+-of-\d+)?\.bin', f)]

    def weight_bytes(self):
        """Stored tensor size, including quantization metadata; runtime is estimated separately."""
        files = self.files()
        weights = self.weight_files()
        index = 'model.safetensors.index.json' if any(f.endswith('.safetensors') for f in weights) else 'pytorch_model.bin.index.json'
        if index in files:
            size = self.json(index).get('metadata', {}).get('total_size')
            if size and int(size) > 0:
                return int(size)
        sizes = [self.file_size(f) for f in weights]
        if not sizes or not all(sizes):
            raise InspectionError('Cannot determine checkpoint weight size')
        return sum(sizes)

    def tensor_headers(self):
        tensors = {}
        files = self.weight_files()
        if not files or len(files) > 64:
            raise InspectionError('Missing or unsupported checkpoint layout')
        for filename in files:
            if filename.endswith('.bin'):
                from flexsds.core.torch_checkpoint_metadata import read_torch_headers
                headers = read_torch_headers(self, filename, self.file_size(filename))
                if tensors.keys() & headers.keys():
                    raise InspectionError('Duplicate checkpoint tensors across shards')
                tensors.update(headers)
                continue
            prefix = self.read(filename, 8, (0, 7))
            size = struct.unpack('<Q', prefix)[0]
            if not 0 < size <= MAX_HEADER:
                raise InspectionError('Invalid safetensors header size')
            header = json.loads(self.read(filename, size, (8, 7 + size)))
            for name, meta in header.items():
                if name != '__metadata__':
                    if name in tensors:
                        raise InspectionError(f'Duplicate checkpoint tensor: {name}')
                    tensors[name] = dict(meta, file=filename, data_start=8 + size)
        return tensors

    def tensor_bytes(self, meta, limit=2 * 1024 * 1024):
        if meta.get('format') == 'pytorch_zip':
            raise InspectionError('Reduced-vocabulary mapping inspection requires safetensors; convert this checkpoint first')
        start, end = meta['data_offsets']
        count = end - start
        if not 0 < count <= limit:
            raise InspectionError('Vocabulary mapping exceeds inspection size limit')
        return self.read(meta['file'], count,
                         (meta['data_start'] + start, meta['data_start'] + end - 1))

    def tokenizer_signature(self):
        """Compare actual token IDs and tokenizer processing, not vocab_size alone."""
        tokenizer = self.json('tokenizer.json')
        relevant = {k: tokenizer.get(k) for k in
                    ('model', 'added_tokens', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder')}
        if not isinstance(tokenizer.get('model'), dict) or not tokenizer['model'].get('vocab'):
            raise InspectionError('Tokenizer has no token-to-ID vocabulary')
        return hashlib.sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
