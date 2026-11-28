"""Tests for CredentialsManager and flexsds auth commands."""
import os
import stat
import pytest
from flexsds.core.credentials import CredentialsManager, mask_secret


def test_credentials_manager_file_permissions_and_masking(tmp_path):
    cred_file = tmp_path / ".flexsds" / "credentials"
    mgr = CredentialsManager(credentials_path=str(cred_file))
    mgr.set_credential("modal", "MODAL_TOKEN_ID", "ak-1234567890abcdef")
    mgr.set_credential("modal", "MODAL_TOKEN_SECRET", "as-9876543210fedcba")

    # Kiểm tra quyền hạn tệp bắt buộc là 0600 (chỉ user đọc/ghi)
    assert cred_file.exists()
    mode = stat.S_IMODE(os.stat(cred_file).st_mode)
    if os.name == "posix":
        assert mode == 0o600
    else:
        # Windows stat modes do not represent ACL confidentiality.
        assert cred_file.read_text(encoding="utf-8")

    # Kiểm tra cơ chế che dấu
    masked = mgr.get_all_masked()
    assert "modal" in masked
    assert "MODAL_TOKEN_ID" in masked["modal"]
    assert masked["modal"]["MODAL_TOKEN_ID"].startswith("ak-")
    assert "1234567890" not in masked["modal"]["MODAL_TOKEN_ID"]
    assert masked["modal"]["MODAL_TOKEN_ID"].endswith("cdef")


def test_credentials_manager_has_no_environment_injection():
    # Dead code removed: preflight reads the saved Hugging Face token directly (hugging_face_token).
    assert not hasattr(CredentialsManager, "inject_into_environ")


def test_credentials_delete_and_logout(tmp_path):
    cred_file = tmp_path / ".flexsds" / "credentials"
    mgr = CredentialsManager(credentials_path=str(cred_file))
    mgr.set_credential("huggingface", "HF_TOKEN", "hf_secret_123456")
    assert mgr.get_credential("huggingface", "HF_TOKEN") == "hf_secret_123456"

    deleted = mgr.delete_credential("huggingface")
    assert deleted is True
    assert mgr.get_credential("huggingface", "HF_TOKEN") is None


def test_mask_secret_helper():
    assert mask_secret("short") == "*****"
    masked = mask_secret("hf_abcdef1234567890")
    assert masked.startswith("hf_")
    assert masked.endswith("7890")
    assert "abcdef" not in masked


def test_cli_auth_status_and_logout(monkeypatch, capsys, tmp_path):
    from flexsds.cli.main import main
    cred_file = tmp_path / ".flexsds" / "credentials"
    monkeypatch.setattr("flexsds.core.credentials.CredentialsManager.DEFAULT_PATH", cred_file)

    mgr = CredentialsManager(str(cred_file))
    mgr.set_credential("modal", "MODAL_TOKEN_ID", "ak-test12345678")

    # Run `flexsds auth status`
    ret = main(["auth", "status"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "modal" in captured.out
    assert "ak-" in captured.out

    # Run `flexsds auth logout --provider modal`
    ret = main(["auth", "logout", "--provider", "modal"])
    assert ret == 0
    assert mgr.get_credential("modal", "MODAL_TOKEN_ID") is None
