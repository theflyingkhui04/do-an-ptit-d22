"""Modal entrypoint for CLI calibration. No serving endpoint is deployed."""
import json
import os
from pathlib import Path
import sys

# The Modal CLI may live in a different pipx environment from flexsds.
parents = Path(__file__).resolve().parents
package_root = str(parents[2]) if len(parents) > 2 else '/root/project/src'
if package_root not in sys.path:
    sys.path.insert(0, package_root)
import modal

request = json.loads(os.environ['FLEXSDS_CALIBRATION_REQUEST'])
config = request['config']
if config.get('target_gpu') not in {'A10G', 'L4', 'T4'}:
    raise ValueError('Unsupported calibration GPU')
app = modal.App('flexsds-calibration')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('vllm==0.30.0', 'psutil')
         .env({'PYTHONPATH': '/root/project/src', 'VLLM_USE_V2_MODEL_RUNNER': '0',
               'FLEXSDS_CALIBRATION_REQUEST': json.dumps(request)}))
if modal.is_local():
    image = image.add_local_dir(str(Path(__file__).resolve().parents[1]),
                                remote_path='/root/project/src/flexsds')
volume = modal.Volume.from_name('ptit-model-cache', create_if_missing=True)


@app.function(image=image, gpu=config['target_gpu'], timeout=21600,
              volumes={'/root/.cache/huggingface': volume})
def measure(settings, batches):
    from flexsds.core.calibration import KS, run_local_group
    return [run_local_group(settings, k, batches) for k in KS]


@app.local_entrypoint()
def main():
    result = measure.remote(config, request['batches'])
    Path(os.environ['FLEXSDS_CALIBRATION_OUTPUT']).write_text(json.dumps(result, allow_nan=False))
