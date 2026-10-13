#!/usr/bin/env python3
"""Preserve four known failed-start R2/R3 cells under an audit-safe suffix.

No deletion and no overwrite; only failed startup with NO probes may be archived.
Preview by default. --archive explicitly permits renaming after full validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / 'experiments/runs/m2_03/diagnostics'
SUFFIX = '.failed_prestart_missing_lane3'
INDEX_NAME = 'm2_03_replication_failed_prestart_archive_index.json'


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def candidates() -> list[tuple[Path, Path]]:
    cells = [('r2', 'on'), ('r2', 'off'), ('r3', 'off'), ('r3', 'on')]
    return [(DIAGNOSTICS / f'k4_cache_{cond}_coldstart_w3_004_{rep}',
             DIAGNOSTICS / f'k4_cache_{cond}_coldstart_w3_004_{rep}{SUFFIX}')
            for rep, cond in cells]


def plan() -> list[dict]:
    index = DIAGNOSTICS / INDEX_NAME
    if index.exists():
        raise RuntimeError(f'STOP: archive index already exists: {index}')
    rows = []
    for src, dst in candidates():
        if not src.is_dir():
            raise RuntimeError(f'STOP: failed-start directory not found: {src}')
        if dst.exists():
            raise RuntimeError(f'STOP: archive destination already exists: {dst}')
        state_file = src / 'run_state.json'
        startup = src / 'startup_stdout.log'
        if not state_file.is_file() or not startup.is_file():
            raise RuntimeError(f'STOP: missing failure metadata: {src}')
        state = json.loads(state_file.read_text(encoding='utf-8'))
        if state.get('status') != 'FAILED_OR_INTERRUPTED':
            raise RuntimeError(f'STOP: not a failed cell: {src}')
        if 'server start failed: exit=1' not in state.get('reason', ''):
            raise RuntimeError(f'STOP: unexpected failure reason: {src}: {state.get("reason")}')
        usage = startup.read_text(encoding='utf-8', errors='replace')
        if 'lane1|lane2|lane3|stop|status' not in usage:
            raise RuntimeError(f'STOP: switch usage not confirmed: {src}')
        for name in ('probes.jsonl', 'all_output_trace.jsonl', 'step_trace.jsonl',
                     'telemetry.jsonl', 'validation.json', 'effective_server_config.json',
                     'probe_stdout.log'):
            if (src / name).exists():
                raise RuntimeError(f'STOP: non-prestart evidence exists: {src / name}')
        files = sorted(p for p in src.rglob('*') if p.is_file())
        if not files:
            raise RuntimeError(f'STOP: unexpectedly empty failure evidence: {src}')
        rows.append({'original_path': str(src.relative_to(ROOT)),
                     'archive_path': str(dst.relative_to(ROOT)),
                     'failure_status': state['status'],
                     'failure_reason': state['reason'],
                     'files_sha256': {str(p.relative_to(src)): digest(p) for p in files}})
    return rows


def archive() -> None:
    rows = plan()  # Validate ALL cells before any filesystem mutation
    print('VALIDATED_FAILED_PRESTART_CELLS:', len(rows))
    for row in rows:
        print('  ', row['original_path'], '->', row['archive_path'])
    completed = []
    try:
        for src, dst in candidates():
            src.rename(dst)  # Preserve all files; do not delete or overwrite
            completed.append((src, dst))
        # Verify byte-for-byte content integrity after renaming.
        for row in rows:
            dst = ROOT / row['archive_path']
            for relative, sha in row['files_sha256'].items():
                if digest(dst / relative) != sha:
                    raise RuntimeError(f'STOP: archive hash mismatch: {dst / relative}')
        report = {'kind': 'failed_prestart_recovery',
                  'reason': 'runner omitted required switch_lane positional argument lane3',
                  'archived_utc': datetime.now(timezone.utc).isoformat(),
                  'archived_without_overwrite': True,
                  'cells': rows}
        target = DIAGNOSTICS / INDEX_NAME
        with target.open('x', encoding='utf-8') as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
            fp.write('\n')
    except BaseException:
        # Best-effort rollback of only our completed renames.
        for src, dst in reversed(completed):
            if dst.exists() and not src.exists():
                dst.rename(src)
        raise
    print('FAILED_PRESTART_EVIDENCE_ARCHIVED: PASS')
    print('AUDIT_INDEX:', DIAGNOSTICS / INDEX_NAME)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', action='store_true', help='Explicitly rename verified failures')
    args = p.parse_args()
    if args.archive:
        archive()
    else:
        rows = plan()
        print('DRY_RUN_ONLY: no files modified; verified failed-start cells:', len(rows))
        for x in rows:
            print('  ', x['original_path'], '->', x['archive_path'],
                  'files:', len(x['files_sha256']))
        print('To preserve these directories under audit suffixes, use --archive')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        raise SystemExit(str(e)) from e
