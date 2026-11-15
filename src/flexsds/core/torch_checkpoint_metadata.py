"""Read torch ZIP tensor metadata without importing torch or loading tensor data.

The restricted decoder only constructs inert storage/tensor descriptors and
OrderedDict objects. Model-provided globals and executable reducers are rejected.
"""
from __future__ import annotations

from collections import OrderedDict
import io
import math
import pickle
import zipfile

LIMIT = 8 * 1024 * 1024
STORAGE_DTYPES = {
    'DoubleStorage': ('F64', 8), 'FloatStorage': ('F32', 4),
    'HalfStorage': ('F16', 2), 'BFloat16Storage': ('BF16', 2),
    'LongStorage': ('I64', 8), 'IntStorage': ('I32', 4),
    'ShortStorage': ('I16', 2), 'CharStorage': ('I8', 1),
    'ByteStorage': ('U8', 1), 'BoolStorage': ('BOOL', 1),
}


class RangeReader(io.RawIOBase):
    def __init__(self, files, filename, size):
        self.files, self.filename, self.size = files, filename, size
        self.position = 0
        self.total_read = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if position < 0:
            raise ValueError('Invalid checkpoint seek')
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b''
        self.total_read += size
        if size > LIMIT or self.total_read > LIMIT * 3:
            raise ValueError('Checkpoint ZIP metadata exceeds inspection budget')
        raw = self.files.read(self.filename, size, (self.position, self.position + size - 1))
        if len(raw) != size:
            raise ValueError('Truncated checkpoint ZIP metadata')
        self.position += size
        return raw


def _tensor(storage, offset, size, stride, *unused):
    if not isinstance(storage, dict) or storage.get('kind') != 'storage':
        raise ValueError('Unsupported checkpoint storage metadata')
    shape = list(size)
    if (len(shape) > 16 or any(type(dim) is not int or dim < 0 for dim in shape)
            or type(offset) is not int or offset < 0 or len(stride) != len(shape)
            or any(type(s) is not int or s < 0 for s in stride)):
        raise ValueError('Invalid checkpoint tensor dimensions')
    extent = offset + sum((dim - 1) * step for dim, step in zip(shape, stride) if dim)
    if math.prod(shape) and extent >= storage['numel']:
        raise ValueError('Checkpoint tensor exceeds declared storage')
    return dict(shape=shape, dtype=storage['dtype'], format='pytorch_zip')


def _parameter(tensor, *unused):
    if not isinstance(tensor, dict) or tensor.get('format') != 'pytorch_zip':
        raise ValueError('Unsupported parameter metadata')
    return tensor


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if module == 'torch' and name in STORAGE_DTYPES:
            return ('storage_type', *STORAGE_DTYPES[name])
        if module == 'torch._utils':
            if name in {'_rebuild_tensor', '_rebuild_tensor_v2'}:
                return _tensor
            if name == '_rebuild_parameter':
                return _parameter
        raise ValueError(f'Unsupported checkpoint metadata global: {module}.{name}')

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != 'storage':
            raise ValueError('Unsupported persistent checkpoint object')
        _, storage_type, key, location, numel = pid
        if (not isinstance(storage_type, tuple) or len(storage_type) != 3
                or storage_type[0] != 'storage_type' or type(numel) is not int or numel < 0):
            raise ValueError('Invalid persistent storage metadata')
        return dict(kind='storage', dtype=storage_type[1], numel=numel)


def _read_torch_headers(files, filename, size):
    with RangeReader(files, filename, size) as source, zipfile.ZipFile(source) as archive:
        members = [entry for entry in archive.infolist() if entry.filename.endswith('/data.pkl') or entry.filename == 'data.pkl']
        if len(members) != 1:
            raise ValueError('Unsupported torch checkpoint ZIP layout')
        member = members[0]
        if member.compress_type != zipfile.ZIP_STORED or not 0 < member.file_size <= LIMIT:
            raise ValueError('Unsupported or oversized checkpoint metadata entry')
        raw = archive.read(member)
    data = MetadataUnpickler(io.BytesIO(raw)).load()
    if isinstance(data, dict) and isinstance(data.get('state_dict'), dict):
        data = data['state_dict']
    if not isinstance(data, dict):
        raise ValueError('Checkpoint does not contain a state dictionary')
    tensors = {name: dict(value, file=filename) for name, value in data.items()
               if isinstance(name, str) and isinstance(value, dict) and value.get('format') == 'pytorch_zip'}
    if not tensors:
        raise ValueError('Checkpoint has no recognized tensor metadata')
    return tensors


def read_torch_headers(files, filename, size):
    try:
        return _read_torch_headers(files, filename, size)
    except (pickle.UnpicklingError, EOFError, zipfile.BadZipFile, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError(f'Invalid torch checkpoint metadata: {exc}') from exc
