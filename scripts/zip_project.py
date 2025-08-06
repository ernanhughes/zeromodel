import os
import zipfile

EXCLUDE_DIRS = {"venv", "__pycache__", ".pytest_cache", ".git", ".github", "docs", "examples", "build", "dist"}
EXCLUDE_SUFFIXES = (".pyc", ".egg-info", ".zip", ".tar.gz", ".whl", ".pyo", ".pyd", ".png")
EXCLUDE_FILES = (".DS_Store", "Thumbs.db", "README.md", "setup.py", "requirements.txt", "LICENSE")

def should_exclude(path):
    parts = path.split(os.sep)
    return any(part in EXCLUDE_DIRS or part.endswith(EXCLUDE_SUFFIXES) for part in parts)

def zip_project_directory(source_dir: str, output_filename: str = "zeromodel.zip", subfolder: str = None):
    """
    Zips a directory while excluding specified folders and files.
    
    Args:
        source_dir (str): The base directory.
        output_filename (str): The name of the output zip file.
        subfolder (str): Optional subdirectory inside source_dir to zip instead of the whole directory.
    """
    root_dir = os.path.join(source_dir, subfolder) if subfolder else source_dir

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Remove excluded directories from the walk
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(EXCLUDE_SUFFIXES)]
            for file in files:
                if file in EXCLUDE_FILES or file.endswith(EXCLUDE_SUFFIXES):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)
                if should_exclude(rel_path):
                    continue
                zipf.write(full_path, rel_path)
    print(f"✅ Project zipped to {output_filename}")

# Example usage
if __name__ == "__main__":
    # To zip everything: zip_project_directory(".")
    # To zip only a subfolder (e.g. "zeromodel"): zip_project_directory(".", subfolder="zeromodel")
    # zip_project_directory(".")
    zip_project_directory(".", subfolder="zeromodel")
