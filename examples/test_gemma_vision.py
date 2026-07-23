import base64
from pathlib import Path
from ollama import chat, show

MODEL = "qwen3.5:latest"

IMAGE_A = Path(
    r"c:\Projects\zeromodel\local-results\gemma4-zero-arcade-20260723T124753Z"
    r"\images\003-tank-0__target-0__cooldown-0.png"
)

IMAGE_B = Path(
    r"c:\Projects\zeromodel\local-results\gemma4-zero-arcade-20260723T124753Z"
    r"\images\006-tank-0__target-6__cooldown-0.png"
)


def encode_image_to_base64(path: Path) -> str:
    """Reads an image file and converts it to a Base64-encoded string."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)

    # Encode image file to base64 string
    base64_image = encode_image_to_base64(path)
    print(f"Encoded image {path} to base64 string of length {len(base64_image)}")

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe only what is visibly present. "
                    "State the cyan tank lane, the magenta target lane, "
                    "and whether the status light is green or red."
                ),
                # Pass the base64-encoded string here
                "images": [base64_image],
            }
        ],
        options={
            "temperature": 0,
            "seed": 0,
        },
    )

    return response.message.content


def main() -> None:
    model_info = show(MODEL)

    print("MODEL INFORMATION")
    print(model_info)
    print()

    print("IMAGE A")
    print(describe(IMAGE_A))
    print()

    print("IMAGE B")
    print(describe(IMAGE_B))


if __name__ == "__main__":
    main()