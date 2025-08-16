import hashlib
import logging

from PIL import Image

from zeromodel.images.vpf import create_vpf, embed_vpf, extract_vpf, verify_vpf

logger = logging.getLogger(__name__)

def test_hash_proof():
    sha3 = lambda b: hashlib.sha3_256(b).hexdigest()

    # 1) Make a tiny artifact (any image works)
    img = Image.new("RGB", (128, 128), (8, 8, 8))

    # 2) Minimal fingerprint (the content hash is filled in during embed)
    vpf = create_vpf(
        pipeline={"graph_hash": "sha3:demo", "step": "render_tile"},
        model={"id": "demo", "assets": {}},
        determinism={"seed": 123, "rng_backends": ["numpy"]},
        params={"size": [128, 128]},
        inputs={"prompt_sha3": sha3(b"hello")},
        metrics={"quality": 0.99},
        lineage={"parents": []},
    )

    # 3) Embed → PNG bytes with footer
    png_with_footer = embed_vpf(img, vpf, mode="stripe")

    # 4) Strip footer to get the core PNG; recompute its SHA3
    idx = png_with_footer.rfind(b"ZMVF")
    core_png = png_with_footer[:idx]
    core_sha3 = "sha3:" + sha3(core_png)

    # 5) Extract fingerprint and verify
    vpf_out, _ = extract_vpf(png_with_footer)
    logger.info("core_sha3         : %s", core_sha3)
    logger.info("fingerprint_sha3  : %s", vpf_out["lineage"]["content_hash"])
    logger.info("verification_pass : %s", verify_vpf(vpf_out, png_with_footer))
