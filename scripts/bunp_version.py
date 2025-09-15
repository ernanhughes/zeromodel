#!/usr/bin/env python3
"""
Bump ZeroModel version.

Usage:
  bump_version.py [major|minor|patch]

Examples:
  bump_version.py patch    # 1.0.4 → 1.0.5
  bump_version.py minor    # 1.0.4 → 1.1.0
  bump_version.py major    # 1.0.4 → 2.0.0
"""

import re
import sys
from pathlib import Path

def get_current_version():
    init_file = Path("zeromodel/__init__.py")
    with open(init_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return None

def bump_version(current, level):
    major, minor, patch = map(int, current.split("."))
    
    if level == "patch":
        patch += 1
    elif level == "minor":
        minor += 1
        patch = 0
    elif level == "major":
        major += 1
        minor = 0
        patch = 0
    
    return f"{major}.{minor}.{patch}"

def update_files(new_version):
    # Update __init__.py
    init_file = Path("zeromodel/__init__.py")
    with open(init_file) as f:
        content = f.read()
    
    content = re.sub(r'__version__\s*=\s*["\'].*?["\']', 
                    f'__version__ = "{new_version}"', content)
    
    with open(init_file, "w") as f:
        f.write(content)
    
    # Update setup.py
    setup_file = Path("setup.py")
    with open(setup_file) as f:
        content = f.read()
    
    content = re.sub(r'version\s*=\s*"1\.\d+\.\d+"', 
                    f'version="{new_version}"', content)
    
    with open(setup_file, "w") as f:
        f.write(content)
    
    print(f"Version updated to {new_version}")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print(__doc__)
        sys.exit(1)
    
    current = get_current_version()
    if not current:
        print("Could not find current version")
        sys.exit(1)
    
    new_version = bump_version(current, sys.argv[1])
    update_files(new_version)