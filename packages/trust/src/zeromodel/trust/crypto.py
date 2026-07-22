"""Ed25519 signing and verification.

Uses the `cryptography` package (a standard, audited library) - no custom
cryptographic primitives are implemented here.

Production artifact loading only ever calls `verify_signature`. Key
generation and signing are authoring/test-only concerns: nothing in this
module writes private key material to disk, a fixture, or a report. Callers
are responsible for keeping any `Ed25519PrivateKey` they generate in memory
only.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from zeromodel.core.artifact import VPMValidationError


@dataclass(frozen=True)
class GeneratedSigningKey:
    """An in-memory keypair for tests/authoring workflows.

    `private_key` is intentionally NOT serializable through this DTO - only
    the public key material (safe to publish) is exposed as hex. Callers
    that need to sign hold on to `private_key` themselves, in memory, for
    the lifetime of the test/script; nothing here ever persists it.
    """

    private_key: Ed25519PrivateKey
    public_key_hex: str


def generate_signing_key() -> GeneratedSigningKey:
    """Authoring/test helper: generate a new in-memory Ed25519 keypair."""
    private_key = Ed25519PrivateKey.generate()
    public_key_hex = _public_key_to_hex(private_key.public_key())
    return GeneratedSigningKey(private_key=private_key, public_key_hex=public_key_hex)


def sign_digest(private_key: Ed25519PrivateKey, digest: str) -> str:
    """Authoring/test helper: sign a `sha256:...` digest string, return hex."""
    if not isinstance(digest, str) or not digest:
        raise VPMValidationError("digest to sign must be a non-empty string")
    signature = private_key.sign(digest.encode("utf-8"))
    return signature.hex()


def verify_signature(*, public_key_hex: str, digest: str, signature_hex: str) -> bool:
    """Production concern: verify a signature over a digest string.

    Returns False (never raises) for any malformed input or invalid
    signature, so callers can treat this as a pure boolean decision input.
    """
    try:
        public_key = _public_key_from_hex(public_key_hex)
        signature = bytes.fromhex(signature_hex)
    except (ValueError, TypeError):
        return False
    try:
        public_key.verify(signature, digest.encode("utf-8"))
    except InvalidSignature:
        return False
    return True


def _public_key_to_hex(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
    return raw.hex()


def _public_key_from_hex(public_key_hex: str) -> Ed25519PublicKey:
    raw = bytes.fromhex(public_key_hex)
    return Ed25519PublicKey.from_public_bytes(raw)
