#!/usr/bin/env python3
"""
Convert markdown tables to PNG images.

Parses markdown files for tables, renders each as a styled PNG using
matplotlib. Outputs one image per table, named by the source file and
table index.

Usage:
    # Single file:
    python analysis_scripts/md_tables_to_images.py analysis_scripts/output/markdown/hardness_error_analysis.md

    # All markdown files in a directory:
    python analysis_scripts/md_tables_to_images.py analysis_scripts/output/markdown/*.md

    # Custom output directory:
    python analysis_scripts/md_tables_to_images.py analysis_scripts/output/markdown/*.md -o analysis_scripts/output/table_images/
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_md_tables(filepath: str) -> list[dict]:
    """Extract tables from a markdown file. Returns list of {title, headers, rows}."""
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    tables = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("|") and "|" in line[1:]:
            # Look back for a title (heading or bold line above the table)
            title = ""
            for j in range(i - 1, max(i - 4, -1), -1):
                candidate = lines[j].strip()
                if candidate.startswith("#"):
                    title = candidate.lstrip("#").strip()
                    break
                if candidate.startswith("*") and candidate.endswith("*"):
                    title = candidate.strip("*").strip()
                    break
                if candidate and not candidate.startswith("|"):
                    title = candidate
                    break

            # Parse header row
            headers = [c.strip() for c in line.split("|")[1:-1]]

            # Skip separator row
            i += 1
            if i < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[i].strip()):
                i += 1

            # Parse data rows
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].split("|")[1:-1]]
                rows.append(cells)
                i += 1

            # Look for a description (italic text or plain text between title and table)
            description = ""
            for j in range(i - len(rows) - 2, max(i - len(rows) - 5, -1), -1):
                candidate = lines[j].strip()
                if candidate.startswith("#") or candidate.startswith("|") or not candidate:
                    continue
                if candidate == title:
                    continue
                description = candidate.strip("*").strip("_").strip()
                break

            if rows:
                tables.append({"title": title, "headers": headers, "rows": rows,
                               "description": description})
        else:
            i += 1

    return tables


DESCRIPTION_FALLBACKS = {
    "Accuracy by Aggressors Count": "Model accuracy grouped by the number of aggressors in the scene. Delta shows the drop from 0 aggressors to 3+.",
    "Accuracy by Victims Count": "Model accuracy grouped by the number of victims in the scene. Delta shows the drop from 0 victims to 3+.",
    "Accuracy by Bystanders Count": "Model accuracy grouped by the number of bystanders in the scene. More bystanders increase visual complexity.",
    "Accuracy by Total People Count": "Model accuracy grouped by the total number of people visible. Performance degrades 10-35pp as scenes get more crowded.",
    "Correct Answer Position Distribution": "Distribution of which letter position (A-H) holds the correct answer. Chance level is 12.5% per position.",
    "Most Frequent Correct Answer Texts": "The most common correct answer strings across the benchmark. High frequency could enable text-only shortcuts.",
    "Model vs Text Frequency Correlation": "Whether models tend to pick the most or least frequent answer text. Similar rates suggest no text-frequency exploitation.",
    "InternVL2.5-78B-AWQ": "Per-question-type breakdown of which distractor types fool InternVL2.5-78B-AWQ when it answers incorrectly.",
    "InternVL2.5-8B": "Per-question-type breakdown of which distractor types fool InternVL2.5-8B when it answers incorrectly.",
    "Qwen3-VL-8B": "Per-question-type breakdown of which distractor types fool Qwen3-VL-8B when it answers incorrectly.",
    "InternVL3.5-8B-DoT": "Per-question-type breakdown of which distractor types fool InternVL3.5-8B-DoT when it answers incorrectly.",
    "Ovis2.5-9B-Thinking": "Per-question-type breakdown of which distractor types fool Ovis2.5-9B-Thinking when it answers incorrectly.",
}


def render_table(table: dict, output_path: Path, max_col_width: int = 22):
    """Render a single table as a PNG image."""
    headers = table["headers"]
    rows = table["rows"]
    title = table["title"]
    n_cols = len(headers)
    n_rows = len(rows)

    def truncate(s, w):
        return (s[:w-2] + "..") if len(s) > w else s

    col_widths = []
    for j in range(n_cols):
        col_vals = [headers[j]] + [rows[i][j] if j < len(rows[i]) else "" for i in range(n_rows)]
        w = min(max(len(v) for v in col_vals) + 2, max_col_width)
        col_widths.append(w)

    fig_width = max(sum(col_widths) * 0.15, 4)
    row_height = 0.3
    fig_height = (n_rows + 1) * row_height + (0.5 if title else 0.15)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    for row in rows:
        padded = []
        for j in range(n_cols):
            val = row[j] if j < len(row) else ""
            val = val.replace("**", "").replace("*", "")
            padded.append(truncate(val, max_col_width))
        cell_text.append(padded)

    clean_headers = [h.replace("**", "").replace("*", "") for h in headers]
    clean_headers = [truncate(h, max_col_width) for h in clean_headers]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=clean_headers,
        cellLoc="center",
        loc="upper center",
    )

    tbl.auto_set_font_size(False)
    font_size = 9 if n_cols <= 6 else 7.5 if n_cols <= 8 else 6.5
    tbl.set_fontsize(font_size)

    header_color = "#4472C4"
    alt_row_color = "#D9E2F3"
    white = "#FFFFFF"

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        cell.set_linewidth(0.5)

        cell_h = 1.0 / (n_rows + 2)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell_h)
        else:
            cell.set_facecolor(alt_row_color if row_idx % 2 == 0 else white)
            cell.set_height(cell_h)

        if col_idx == 0 and row_idx > 0:
            cell.set_text_props(ha="left")

    tbl.auto_set_column_width(list(range(n_cols)))

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    description = table.get("description", "")
    if not description:
        for key, fallback in DESCRIPTION_FALLBACKS.items():
            if (title or "") == key or (title or "").startswith(key + " "):
                description = fallback
                break

    if description:
        fig.text(0.5, -0.02, description, ha="center", va="top",
                 fontsize=8, fontstyle="italic", color="#444444",
                 wrap=True, transform=fig.transFigure)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close()

    from PIL import Image as PILImage
    img = PILImage.open(output_path)
    bg = PILImage.new(img.mode, img.size, (255, 255, 255))
    diff = list(img.getdata())
    w, h = img.size
    bottom = h
    for y in range(h - 1, 0, -1):
        row = diff[y * w:(y + 1) * w]
        if any(p != (255, 255, 255) and p != (255, 255, 255, 255) for p in row):
            bottom = y + 2
            break
    if bottom < h:
        img = img.crop((0, 0, w, min(bottom + 10, h)))
        img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Markdown files to process")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: same directory as input)")
    args = parser.parse_args()

    total = 0
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"  SKIP {filepath}: not found")
            continue

        tables = parse_md_tables(filepath)
        if not tables:
            print(f"  SKIP {filepath}: no tables found")
            continue

        out_dir = Path(args.output_dir) if args.output_dir else path.parent / "table_images"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = path.stem
        for idx, table in enumerate(tables):
            suffix = f"_{idx+1}" if len(tables) > 1 else ""
            out_path = out_dir / f"{stem}{suffix}.png"
            render_table(table, out_path)
            title_preview = table['title'][:50] if table['title'] else '(untitled)'
            print(f"  {out_path}: {title_preview}")
            total += 1

    print(f"\nGenerated {total} table images")


if __name__ == "__main__":
    main()
