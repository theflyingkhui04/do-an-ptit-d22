"""The web console's sample `flexsds serve` command must be one the real CLI accepts."""
import re
import shlex
from pathlib import Path

from flexsds.cli.main import parse_args

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web/src/components/EndpointsPage.tsx"


def web_constant(name: str) -> str:
    match = re.search(rf'export const {name} =\s*"([^"]+)"', PAGE.read_text(encoding="utf-8"))
    assert match, f"{name} not found in {PAGE.name}"
    return match.group(1)


def test_sample_command_parses_with_the_real_cli():
    argv = shlex.split(web_constant("SERVE_SAMPLE"))
    assert argv[:2] == ["flexsds", "serve"]
    args = parse_args(argv[1:])  # argparse exits with code 2 on an unknown flag such as --model
    assert args.command == "serve"
    assert args.target and args.draft


def test_sample_output_matches_what_the_cli_prints():
    session = (ROOT / "src/flexsds/cli/serving_session.py").read_text(encoding="utf-8")
    header = web_constant("SERVE_SAMPLE_HEADER")
    assert f"'\\n{header}'" in session
    app_name = re.search(r'APP_NAME = "([^"]+)"',
                         (ROOT / "src/flexsds/server/modal_app.py").read_text(encoding="utf-8")).group(1)
    assert web_constant("SERVE_SAMPLE_API").endswith(f"--{app_name}-serve.modal.run/v1")
