def test_vpf_deterministic_replay():
    """
    Generate an artifact with a VPF, then replay it later and verify
    that the regenerated output matches bit-for-bit.
    """
    from zeromodel.images import embed_vpf, extract_vpf, replay_from_vpf
    import hashlib

    # Step 1: Generate an artifact with a VPF
    artifact, vpf = embed_vpf(prompt="cyberpunk cityscape at night")

    # Step 2: Compute content hash
    original_hash = hashlib.sha3_256(artifact).hexdigest()

    # Step 3: Extract VPF later (simulating time gap)
    extracted_vpf = extract_vpf(artifact)

    # Step 4: Replay exactly from VPF
    replayed_artifact = replay_from_vpf(extracted_vpf)

    # Step 5: Verify bit-for-bit identity
    replayed_hash = hashlib.sha3_256(replayed_artifact).hexdigest()
    assert replayed_hash == original_hash, "Replayed artifact does not match original"

    print("✅ Deterministic replay successful — provenance verified")
