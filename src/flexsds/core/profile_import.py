"""Import JSON profiles into the local artifact store and SQLite catalog."""
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlparse
from urllib.request import urlopen
from flexsds.core import cost_model

MAX_BYTES = 16 * 1024 * 1024


def read_profile_source(source):
    source = source.strip().strip('"').strip("'")
    parsed = urlparse(source)
    if parsed.scheme in ('http', 'https'):
        with urlopen(source, timeout=30) as response:
            raw = response.read(MAX_BYTES + 1)
    else:
        path = Path(source).expanduser()
        if path.stat().st_size > MAX_BYTES:
            raise ValueError('Profile file exceeds 16 MiB.')
        raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError('Profile file exceeds 16 MiB.')
    return raw.decode('utf-8-sig')


def inspect_profile(text):
    if len(text.encode('utf-8')) > MAX_BYTES:
        raise ValueError('Profile exceeds 16 MiB.')
    data = json.loads(text.lstrip('\ufeff'))
    if not isinstance(data, dict):
        raise ValueError('Profile must be a JSON object.')
    meta, models = data.get('metadata', {}), data.get('models', {})
    if not isinstance(meta, dict) or not isinstance(models, dict):
        raise ValueError('Profile metadata and models must be objects.')
    if not re.fullmatch(r'sm_\d{2,3}', str(meta.get('sm_architecture', ''))):
        raise ValueError('Profile metadata.sm_architecture must identify a GPU, e.g. sm_75.')
    if not all(isinstance(v, str) and v.strip() for v in
               (meta.get('device'), models.get('target'), models.get('draft'))):
        raise ValueError('Profile requires device, target model, and draft model metadata.')
    # Reject non-finite values even in fields not consumed by a legacy reader.
    json.dumps(data, allow_nan=False)
    if meta.get('schema') == 'flexsds-component-v2':
        cost_model.validate_measured_profile(data)
        status = 'measured'
    elif meta.get('schema'):
        raise ValueError('Unsupported hardware profile schema.')
    else:
        model = cost_model.FlexCostModel.from_json(data)
        values = [*model.matrix.values(), *model.forward_target.values(), *model.forward_draft.values()]
        if not values or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Legacy profile must contain positive finite latency costs.')
        if not model.matrix and not (model.forward_target and model.forward_draft):
            raise ValueError('Legacy profile requires a cost matrix or both forward cost tables.')
        status = 'legacy'
    return data, status


def import_profile(text, directory=None):
    data, status = inspect_profile(text)
    directory = Path(directory) if directory is not None else cost_model.USER_PROFILES_DIR
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    meta, models = data['metadata'], data['models']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"profile_{meta['sm_architecture']}_import_{digest}.json"
    created = not path.exists()
    if not created:
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved != data:
            raise ValueError('Existing profile artifact has different contents; import was not saved.')
    else:
        # Reuse the atomic artifact writer; the content hash avoids overwriting other profiles.
        path = cost_model.save_hardware_profile(data, meta['sm_architecture'],
                                               f'import_{digest}', target_dir=directory)
    try:
        with sqlite3.connect(directory/'profiles.sqlite3') as database:
            database.execute('''CREATE TABLE IF NOT EXISTS profiles (
                sha256 TEXT PRIMARY KEY, filename TEXT NOT NULL, sm_architecture TEXT NOT NULL,
                device TEXT NOT NULL, target_model TEXT NOT NULL, draft_model TEXT NOT NULL,
                status TEXT NOT NULL, imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
            database.execute('''INSERT INTO profiles
                (sha256,filename,sm_architecture,device,target_model,draft_model,status)
                VALUES (?,?,?,?,?,?,?) ON CONFLICT(sha256) DO NOTHING''',
                (digest,path.name,meta['sm_architecture'],meta['device'],models['target'],models['draft'],status))
    except sqlite3.Error:
        if created:
            path.unlink(missing_ok=True)
        raise
    return path, status
