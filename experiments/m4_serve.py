"""Canonical vLLM CLI with a source-gated frontend lifecycle correction."""
import sys


def main():
    from experiments.m4_shutdown import install
    install()
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
