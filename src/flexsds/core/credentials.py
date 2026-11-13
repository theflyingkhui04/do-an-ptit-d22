"""Zero-leak secure credentials manager for FlexSDS CLI."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, Optional


def mask_secret(secret: str) -> str:
    """Masks secret keeping prefix and suffix visible."""
    if not secret or len(secret) <= 8:
        return "*****"
    return f"{secret[:3]}****...{secret[-4:]}"


class CredentialsManager:
    """Manages sensitive API credentials outside repo in user home with 0600 permissions."""

    DEFAULT_PATH = Path.home() / ".flexsds" / "credentials"

    def __init__(self, credentials_path: Optional[str] = None) -> None:
        if credentials_path:
            self.path = Path(credentials_path).expanduser().resolve()
        else:
            self.path = self.DEFAULT_PATH

    def _ensure_dir_and_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass

        if not self.path.exists():
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _load(self) -> Dict[str, Dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: Dict[str, Dict[str, str]]) -> None:
        self._ensure_dir_and_file()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def set_credential(self, provider: str, key: str, value: str) -> None:
        """Stores a credential for a given provider (modal, huggingface, etc.)."""
        data = self._load()
        if provider not in data:
            data[provider] = {}
        data[provider][key] = value
        self._save(data)

    def get_stored_credential(self, provider: str, key: str) -> Optional[str]:
        """Retrieves stored credential exclusively from credentials file."""
        data = self._load()
        return data.get(provider, {}).get(key)

    def get_credential(self, provider: str, key: str, fallback_env: bool = True) -> Optional[str]:
        """Retrieves a credential, checking os.environ first if fallback_env is True."""
        if fallback_env:
            env_val = os.environ.get(key)
            if env_val:
                return env_val
        return self.get_stored_credential(provider, key)

    def delete_credential(self, provider: str) -> bool:
        """Deletes all stored credentials for a given provider and cleans environment."""
        data = self._load()
        if provider in data:
            keys = data[provider]
            for k in keys:
                os.environ.pop(k, None)
            del data[provider]
            self._save(data)
            return True
        return False

    def get_all_masked(self) -> Dict[str, Dict[str, str]]:
        """Returns all stored credentials with masked values for safe display."""
        data = self._load()
        masked: Dict[str, Dict[str, str]] = {}
        for prov, keys in data.items():
            masked[prov] = {}
            for k, v in keys.items():
                masked[prov][k] = mask_secret(v)
        return masked

    def inject_into_environ(self) -> None:
        """Injects stored credentials into os.environ in memory if not already set."""
        data = self._load()
        for _, keys in data.items():
            for k, v in keys.items():
                if k not in os.environ:
                    os.environ[k] = v
