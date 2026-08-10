#!/usr/bin/env python3
import subprocess
import json
from pathlib import Path

ROOT = Path("/scratch/kcwp264/Conditional-GQE_materials")
OUT_DIR = ROOT / "docs" / "mermaid_svgs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Get original README content from git commit 7062390
raw_readme = subprocess.check_output(["git", "show", "7062390:README.md"]).decode("utf-8")

lines = raw_readme.split("\n")
blocks = []
i = 0
while i < len(lines):
    if lines[i].strip().startswith("```mermaid"):
        start = i
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith("```"):
            j += 1
        if j < len(lines):
            content = "\n".join(lines[start+1:j])
            blocks.append(content)
            i = j + 1
        else:
            i += 1
    else:
        i += 1

print(f"Extracted {len(blocks)} Mermaid blocks from git commit 7062390.")

# Puppeteer config
puppeteer_config = {
    "args": ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
}
pup_cfg_path = Path("/tmp/puppeteer-config.json")
pup_cfg_path.write_text(json.dumps(puppeteer_config))

# Mermaid config for high readability, crisp light background, clear dark text & themes
mermaid_config = {
    "theme": "default",
    "themeVariables": {
        "primaryColor": "#e8eaf6",
        "primaryTextColor": "#1a237e",
        "primaryBorderColor": "#3f51b5",
        "lineColor": "#283593",
        "secondaryColor": "#f3e5f5",
        "tertiaryColor": "#e0f2f1",
        "fontSize": "16px"
    },
    "flowchart": {
        "htmlLabels": True,
        "useMaxWidth": True
    }
}
mmd_cfg_path = Path("/tmp/mermaid-config.json")
mmd_cfg_path.write_text(json.dumps(mermaid_config))

for idx, content in enumerate(blocks, start=1):
    mmd_file = OUT_DIR / f"diagram_{idx:02d}.mmd"
    svg_file = OUT_DIR / f"diagram_{idx:02d}.svg"
    png_file = OUT_DIR / f"diagram_{idx:02d}.png"
    
    mmd_file.write_text(content, encoding="utf-8")
    
    # Render PNG directly using mmdc (Puppeteer screenshot preserves all text & styling)
    cmd_png = [
        "npx", "-y", "@mermaid-js/mermaid-cli",
        "-p", str(pup_cfg_path),
        "-c", str(mmd_cfg_path),
        "-i", str(mmd_file),
        "-o", str(png_file),
        "-w", "2000",
        "-b", "white",
        "-s", "2"
    ]
    print(f"Rendering Diagram {idx:02d} to PNG...")
    res_png = subprocess.run(cmd_png, capture_output=True, text=True)
    if res_png.returncode != 0:
        print(f"  PNG Error: {res_png.stderr}")
    else:
        print(f"  Successfully created {png_file.name} ({png_file.stat().st_size} bytes)")

    # Render SVG directly as well
    cmd_svg = [
        "npx", "-y", "@mermaid-js/mermaid-cli",
        "-p", str(pup_cfg_path),
        "-c", str(mmd_cfg_path),
        "-i", str(mmd_file),
        "-o", str(svg_file),
        "-w", "2000",
        "-b", "white"
    ]
    subprocess.run(cmd_svg, capture_output=True, text=True)

print("All diagrams rendered directly to PNG & SVG with full text!")
