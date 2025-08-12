import hashlib

def sha3_bytes(b: bytes) -> str:
    """
    Compute the SHA3-256 hash of input bytes and return as hex string.
    
    This is the core hashing function used throughout ZeroModel for:
    - Content addressing of VPM tiles
    - Provenance verification
    - Deterministic tile identification
    - Cryptographic integrity checks
    
    Args:
        b: Input bytes to hash
        
    Returns:
        Hexadecimal string representation of the SHA3-256 hash
        
    Example:
        >>> sha3_bytes(b"hello world")
        '944ad329d0fc15a38889e8d61a3d8e127506a0c8e67f8a8e1d3d6e9d3d0c6d0c'
    
    Note:
        SHA3-256 is used instead of SHA-256 for better resistance against length
        extension attacks and as part of the ZeroModel's commitment to modern
        cryptographic standards for provenance.
    """
    return hashlib.sha3_256(b).hexdigest()