"""Argon2id password verification for the single admin account."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=2,
    hash_len=32,
    salt_len=16,
)


def verify_admin_password(*, password_hash: str, password: str) -> bool:
    """
    Verify a plaintext password against an Argon2id encoded hash.

    Returns False on mismatch or invalid hash — never raises to callers for
    ordinary credential failures.
    """
    try:
        return bool(_HASHER.verify(password_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:  # noqa: BLE001 — fail closed
        return False


def hash_admin_password(password: str) -> str:
    """Helper for operators generating hashes offline (not used at runtime login)."""
    return _HASHER.hash(password)
