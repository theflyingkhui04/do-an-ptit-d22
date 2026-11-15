"""Dispatch calibration through the Modal CLI, including pipx-isolated installs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from flexsds.cli.process_progress import run_logged_process


def calibrate(config, batches):
    from flexsds.server.modal_runner import ModalServingRunner
    from flexsds.core.calibration import KS
    if config.get('target_gpu') not in {'A10G', 'L4', 'T4'}:
        raise ValueError('Unsupported Modal calibration GPU')
    env = ModalServingRunner().setup_environment({})
    env['FLEXSDS_CALIBRATION_REQUEST'] = json.dumps(dict(config=config, batches=batches))
    with tempfile.TemporaryDirectory(prefix='flexsds-modal-calibration-') as directory:
        output = Path(directory) / 'result.json'
        env['FLEXSDS_CALIBRATION_OUTPUT'] = str(output)
        app_path = str(Path(__file__).with_name('modal_calibration_app.py'))
        try:
            run_logged_process([shutil.which('modal') or 'modal', 'run', app_path],
                               env=env, timeout=21600, total_cells=len(KS)*len(batches))
            return json.loads(output.read_text())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f'Modal GPU calibration failed: {exc}') from exc
