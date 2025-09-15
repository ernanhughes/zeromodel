import os
import re

TARGET_IMPORT = "from __future__ import annotations\n"
HEADER_PREFIX = "#  "  # keep your exact style

ENCODING_RE = re.compile(r"coding[:=]\s*([-\w.]+)")

def _after_shebang_and_encoding(lines):
    """Return index just after shebang and encoding cookie."""
    i = 0
    n = len(lines)
    if i < n and lines[i].startswith("#!"):
        i += 1
    # PEP 263: encoding must be in first or second line
    if i < n and ENCODING_RE.search(lines[i]):
        i += 1
    elif i == 0 and n >= 2 and ENCODING_RE.search(lines[1]):
        i = 2
    return i

def _docstring_span(lines, start_idx=0):
    """
    If a top-level docstring exists starting at/after start_idx (skipping blanks/comments),
    return (start, end_exclusive). Otherwise return None.
    """
    n = len(lines)
    j = start_idx
    # skip blanks and comments
    while j < n and (not lines[j].strip() or lines[j].lstrip().startswith("#")):
        j += 1
    if j >= n:
        return None
    l = lines[j].lstrip()
    if l.startswith(("'''", '"""')):
        quote = "'''" if l.startswith("'''") else '"""'
        # single-line docstring (start and end on same line)
        if l.count(quote) >= 2:
            return (j, j + 1)
        # multi-line: find closing
        k = j + 1
        while k < n:
            if quote in lines[k]:
                return (j, k + 1)
            k += 1
    return None

def _compute_header_text(root_dir, filepath):
    """Build a header like '#  zeromodel/constants.py' regardless of OS."""
    base = os.path.basename(os.path.normpath(root_dir)).replace(os.sep, "/")
    rel = os.path.relpath(filepath, root_dir).replace(os.sep, "/")
    # Always prefix with the root folder name (e.g., 'zeromodel/relpath')
    if base and rel and not rel.startswith(base + "/"):
        shown = f"{base}/{rel}"
    else:
        # Handles cases where root_dir is '.' or already in rel
        shown = f"{base}/{rel}" if base and base not in ("", ".") and not rel.startswith(base + "/") else (rel if rel else base)
    return f"{HEADER_PREFIX}{shown}\n"

def ensure_header_and_future_annotations(root_dir="zeromodel"):
    modified_files = []

    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue

            path = os.path.join(subdir, fname)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            desired_header = _compute_header_text(root_dir, path)
            changed = False

            # 1) find insertion point just after shebang+encoding
            preamble_idx = _after_shebang_and_encoding(lines)

            # 2) ensure/normalize header at that spot (or in first few lines after it)
            #    - if a header exists but doesn't match, replace it
            #    - otherwise insert our desired header
            header_idx = None
            scan_limit = min(preamble_idx + 5, len(lines))
            for i in range(preamble_idx, scan_limit):
                s = lines[i].strip()
                if s.startswith(HEADER_PREFIX.strip()):
                    header_idx = i
                    break

            if header_idx is None:
                # insert header at preamble
                lines.insert(preamble_idx, desired_header)
                changed = True
                header_idx = preamble_idx
            else:
                if lines[header_idx] != desired_header:
                    lines[header_idx] = desired_header
                    changed = True

            # 3) ensure future import AFTER docstring (if any), otherwise after header
            has_future = any("from __future__ import annotations" in l for l in lines)
            if not has_future:
                # Docstring detection begins AFTER the header line we just guaranteed
                doc_span = _docstring_span(lines, start_idx=header_idx + 1)
                insert_at = (doc_span[1] if doc_span else (header_idx + 1))
                # Avoid inserting before another import already present; optional—but safe to keep simple.
                lines.insert(insert_at, TARGET_IMPORT)
                changed = True

            if changed:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                modified_files.append(path)

    if modified_files:
        print("✅ Updated files (header and/or future import):")
        for p in modified_files:
            print("   -", p)
    else:
        print("No changes needed.")

if __name__ == "__main__":
    ensure_header_and_future_annotations(root_dir="zeromodel")
