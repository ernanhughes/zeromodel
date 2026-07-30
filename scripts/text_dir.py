import os
import sys
from pathlib import Path


def recreate_tree(input_file):
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    path_stack = []
    # We use a dictionary to map indentation levels to stack depths
    # This handles mixed 2-space/4-space or tabbed files automatically
    indent_map = {0: 0}

    print(f"🚀 Starting structure recreation from: {input_file}\n" + "-" * 50)

    for i, line in enumerate(lines, 1):
        # Skip truly empty lines or lines that are just tree connectors with no names
        stripped = line.strip()
        if not stripped or stripped in ["│", "├──", "└──"]:
            continue

        try:
            # 1. Calculate indentation based on the first alphanumeric character
            clean_line = (
                line.replace("├──", "    ").replace("└──", "    ").replace("│", " ")
            )
            name = clean_line.strip()
            indent_size = clean_line.find(name)

            # 2. Determine Depth (Smarter Logic)
            # We track which 'indent_size' corresponds to which level in the stack
            sorted_indents = sorted(indent_map.keys())
            if indent_size not in indent_map:
                # Assign the next available depth level
                indent_map[indent_size] = len(sorted_indents)

            depth = indent_map[indent_size]

            # 3. Handle File vs Directory
            is_directory = name.endswith("/") or "." not in name
            actual_name = name.rstrip("/")

            # 4. Update path stack
            path_stack = path_stack[:depth]
            path_stack.append(actual_name)

            target_path = Path(*path_stack)

            # 5. Execution with Error Handling
            if is_directory:
                target_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ DIR  : {target_path}")
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # We use 'touch' but check if it's already a directory first
                if target_path.exists() and target_path.is_dir():
                    print(f"⚠️  SKIP : {target_path} (Path is already a directory)")
                else:
                    target_path.touch(exist_ok=True)
                    print(f"📄 FILE : {target_path}")

        except Exception as e:
            # The "Don't Bail" Logic: Log the error and move to the next line
            print(f"❌ ERROR: Line {i} ('{stripped}') failed. Reason: {e}")
            continue

    print("-" * 50 + "\n✅ Process complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tree_dir.py <structure_file.txt>")
    else:
        recreate_tree(sys.argv[1])
