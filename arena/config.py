"""Runtime configuration: where the backend lives and who you are.

The golden path needs exactly one thing set — your team token. The backend and
data URLs default to the workshop's deployed services, so::

    import arena
    arena.configure(token="wksp_team07_xxxx")

is enough. Everything is overridable by env var (``ARENA_TOKEN``,
``ARENA_BACKEND_URL``, ``ARENA_LEADERBOARD_URL``, ``ARENA_DATA_URL``) for the
cookers running outside the notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Baked defaults for the live workshop services. Overridable via env / configure().
_BAKED_BACKEND_URL = "https://carnation-workshop--cell-arena-models-web.modal.run"
_BAKED_LEADERBOARD_URL = "https://carnation-workshop--cell-arena-leaderboard-web.modal.run"
_BAKED_DATA_URL = "https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data"


@dataclass
class Config:
    token: str
    backend_url: str
    leaderboard_url: str
    data_url: str


_config = Config(
    token=os.environ.get("ARENA_TOKEN", ""),
    backend_url=os.environ.get("ARENA_BACKEND_URL", _BAKED_BACKEND_URL).rstrip("/"),
    leaderboard_url=os.environ.get("ARENA_LEADERBOARD_URL", _BAKED_LEADERBOARD_URL).rstrip("/"),
    data_url=os.environ.get("ARENA_DATA_URL", _BAKED_DATA_URL).rstrip("/"),
)


def configure(
    token: str | None = None,
    backend_url: str | None = None,
    leaderboard_url: str | None = None,
    data_url: str | None = None,
) -> None:
    """Set any of the runtime config values (others are left as-is)."""
    if token is not None:
        _config.token = token
    if backend_url is not None:
        _config.backend_url = backend_url.rstrip("/")
    if leaderboard_url is not None:
        _config.leaderboard_url = leaderboard_url.rstrip("/")
    if data_url is not None:
        _config.data_url = data_url.rstrip("/")


def get_config() -> Config:
    """Return the active configuration."""
    return _config
