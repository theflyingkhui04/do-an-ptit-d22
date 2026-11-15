"""Synthetic torch ZIP metadata fixtures; never import torch or allocate tensors."""
from collections import OrderedDict
import io
import pickle
import sys
import types
import zipfile

import pytest

from flexsds.core.model_inspection import ModelFiles
from flexsds.core.torch_checkpoint_metadata import read_torch_headers


@pytest.fixture
def torch_zip(tmp_path, monkeypatch):
    # These stand-ins exist only to emit the same pickle globals as torch.save.
    # The reader must replace them with inert metadata builders.
    torch = types.ModuleType('torch')
    utils = types.ModuleType('torch._utils')
    exec("def _rebuild_tensor_v2(*args): raise AssertionError('Original reducer executed')", utils.__dict__)
    storage_type = type('BFloat16Storage', (), {'__module__': 'torch'})
    torch.BFloat16Storage = storage_type
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch._utils', utils)
    storage = object()

    class Tensor:
        def __reduce__(self):
            return utils._rebuild_tensor_v2, (storage, 0, (4, 8), (8, 1), False, OrderedDict())

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if obj is storage:
                return ('storage', storage_type, '0', 'cpu', 32)

    data = io.BytesIO()
    Pickler(data, protocol=2).dump(OrderedDict([('fc.weight', Tensor())]))
    path = tmp_path / 'pytorch_model.bin'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', data.getvalue())
        archive.writestr('archive/data/0', b'\0' * 2_000_000)
    return tmp_path, path


def test_reads_tensor_metadata_without_loading_weights(torch_zip, monkeypatch):
    directory, path = torch_zip
    files = ModelFiles(str(directory))
    original = files.read
    requested = []
    def bounded(filename, limit, byte_range):
        requested.append(limit)
        return original(filename, limit, byte_range)
    monkeypatch.setattr(files, 'read', bounded)
    tensors = files.tensor_headers()
    assert tensors['fc.weight']['shape'] == [4, 8]
    assert tensors['fc.weight']['dtype'] == 'BF16'
    assert sum(requested) < 100_000
    assert files.weight_bytes() == path.stat().st_size


def test_pytorch_remote_inspection_requires_only_bounded_ranges(torch_zip, monkeypatch):
    directory, path = torch_zip
    raw = path.read_bytes()
    files = ModelFiles(str(directory))
    ranges = []
    def remote(filename, limit, byte_range):
        start, end = byte_range
        ranges.append((start, end))
        assert limit < 100_000
        return raw[start:end + 1]
    monkeypatch.setattr(files, 'read', remote)
    assert files.tensor_headers()['fc.weight']['shape'] == [4, 8]
    assert len(ranges) < 15


def test_arbitrary_pickle_globals_are_never_executed(tmp_path):
    class Malicious:
        def __reduce__(self):
            return eval, ("(_ for _ in ()).throw(AssertionError('executed'))",)
    path = tmp_path / 'pytorch_model.bin'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', pickle.dumps(Malicious()))
    files = ModelFiles(str(tmp_path))
    with pytest.raises(ValueError, match='metadata global'):
        files.tensor_headers()


def test_corrupt_pytorch_zip_is_a_handled_metadata_error(tmp_path):
    path = tmp_path / 'pytorch_model.bin'
    path.write_bytes(b'bad checkpoint')
    with pytest.raises(ValueError, match='Invalid .*checkpoint'):
        ModelFiles(str(tmp_path)).tensor_headers()


def test_qwen_legacy_head_is_recognized_from_pytorch_metadata(torch_zip):
    import json
    from flexsds.core.speculative_method import inspect_method
    directory, _ = torch_zip
    (directory / 'config.json').write_text(json.dumps(dict(architectures=['Qwen2ForCausalLM'], num_hidden_layers=1, hidden_size=4)))
    detected = inspect_method(str(directory))
    assert detected.method == 'eagle'
    assert detected.verified
