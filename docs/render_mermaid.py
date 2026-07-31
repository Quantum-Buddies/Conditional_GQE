#!/usr/bin/env python3
"""Extract Mermaid blocks from README.md, render to SVG, replace with img tags."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
OUT_DIR = ROOT / "docs" / "mermaid_svgs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_mermaid_blocks(text):
    """Return list of (start_line, end_line, content) for each mermaid block."""
    lines = text.split("\n")
    blocks = []
    i = 0
    idx = 0
    while i < len(lines):
        if lines[i].strip() == "```mermaid":
            start = i
            j = i + 1
            while j < len(lines) and lines[j].strip() != "```":
                j += 1
            if j < len(lines):
                content = "\n".join(lines[start+1:j])
                blocks.append((start, j, content, idx))
                idx += 1
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    return blocks

def render_mermaid_to_svg(content, out_path, width=1600):
    """Render mermaid content to SVG using mmdc."""
    mmd_file = out_path.with_suffix(".mmd")
    mmd_file.write_text(content)
    cmd = [
        "npx", "-y", "@mermaid-js/mermaid-cli",
        "-i", str(mmd_file),
        "-o", str(out_path),
        "-w", str(width),
        "-b", "transparent",
    ]
    print(f"  Rendering {out_path.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  WARNING: mmdc failed for {out_path.name}: {result.stderr[:500]}")
        return False
    mmd_file.unlink(missing_ok=True)
    return True

def main():
    text = README.read_text()
    blocks = extract_mermaid_blocks(text)
    print(f"Found {len(blocks)} Mermaid blocks")

    # Render each block
    svg_paths = []
    for start, end, content, idx in blocks:
        svg_name = f"diagram_{idx+1:02d}.svg"
        svg_path = OUT_DIR / svg_name
        success = render_mermaid_to_svg(content, svg_path)
        svg_paths.append((svg_name, success))

    # Replace mermaid blocks with image references
    lines = text.split("\n")
    # Process from bottom to top so line indices don't shift
    for start, end, content, idx in reversed(blocks):
        svg_name = f"diagram_{idx+1:02d}.svg"
        svg_rel = f"docs/mermaid_svgs/{svg_name}"
        # Replace the entire ```mermaid ... ``` block with an image reference
        replacement = f'<img src="{svg_rel}" alt="Diagram {idx+1}" width="100%">'
        lines[start:end+1] = [replacement]

    new_text = "\n".join(lines)
    README.write_text(new_text)
    print(f"\nUpdated README.md with {len(blocks)} image references")

    # Summary
    succeeded = sum(1 for _, s in svg_paths if s)
    failed = len(svg_paths) - succeeded
    print(f"  Rendered: {succeeded}/{len(blocks)}")
    if failed:
        print(f"  Failed: {failed}")
        for name, s in svg_paths:
            if not s:
                print(f"    - {name}")

if __name__ == "__main__":
    main()
