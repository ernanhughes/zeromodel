import os
import sys

MODEL_TYPE_ICONS = {
    "svm": "�",
    "mrq": "🧠",
    "ebt": "🪜"
}

def get_icon(name, is_dir):
    """Return an emoji icon based on the filename or extension."""
    name = name.lower()
    if is_dir:
        return "📁"
    elif "encoder.pt" in name:
        return "🧠"
    elif "model.pt" in name:
        return "🤖"
    elif "tuner.json" in name:
        return "🎚️"
    elif "_scaler.joblib" in name:
        return "📏"
    elif name.endswith("meta.json"):
        return "⚙️"
    elif name.endswith((".yaml", ".yml")):
        return "📄"
    elif name.endswith(".md"):
        return "📘"
    elif name.endswith(".json"):
        return "🗂️"
    elif name.endswith(".pt"):
        return "📦"
    elif name.endswith(".joblib") or name.endswith(".pkl"):
        return "📦"
    elif name.endswith(".onnx"):
        return "📐"
    elif name.endswith(".txt"):
        return "📝"
    elif name.endswith(".py"):
        return "🐍"
    else:
        return "📦"

def get_model_type_icon(name):
    """Return a model type icon if the folder matches a known model type."""
    lower_name = name.lower()
    return MODEL_TYPE_ICONS.get(lower_name, "📦")

def print_tree(root_path, file_obj=sys.stdout, indent="", is_top_level=True):
    """
    Recursively print the directory tree to a given file object with icons.
    Defaults to printing to stdout.
    """
    try:
        entries = sorted(os.listdir(root_path))
    except PermissionError:
        print(f"{indent}└── 🔒 Permission Denied", file=file_obj)
        return

    for i, entry in enumerate(entries):
        path = os.path.join(root_path, entry)
        if entry.startswith("."):
            continue
        if entry.startswith("__"):
            continue
        is_dir = os.path.isdir(path)
        connector = "└──" if i == len(entries) - 1 else "├──"

        # Use model-type icon at top-level (models/svm, models/mrq, etc.)
        if is_top_level and is_dir:
            icon = get_model_type_icon(entry)
        else:
            icon = get_icon(entry, is_dir)

        print(f"{indent}{connector} {icon}  {entry}", file=file_obj)

        if is_dir:
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, file_obj, indent + extension, is_top_level=False)

if __name__ == "__main__":
    base_dir = "zeromodel"  # Replace with your actual base path
    output_filename = "file_view.log"
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory '{base_dir}' does not exist.", file=sys.stderr)
    else:
        with open(output_filename, "w", encoding="utf-8") as f:
            print(f"📦 {base_dir}", file=f)
            print_tree(base_dir, file_obj=f)
        print(f"✅ Directory tree successfully written to '{output_filename}'")
