#!/usr/bin/env python3
"""Convert layout JSONL results into per-page plain-text files."""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datas"
PAGE_FILE_PATTERN = re.compile(r"^page_(\d+)\.txt$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect MMD OCRAG and ViDoSeek layout text into data/*/layout_ocr."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Target dataset root (default: data/).",
    )
    parser.add_argument(
        "--mmd-layout-dir",
        type=Path,
        default=DATASETS_ROOT / "mmdocrag" / "layout_content",
        help="Directory containing MMD OCRAG *_layout.jsonl files.",
    )
    parser.add_argument(
        "--vidoseek-layout",
        type=Path,
        default=(
            DATASETS_ROOT
            / "vidoseek"
            / "layout_content"
            / "vidoseek_layout.jsonl"
        ),
        help="ViDoSeek layout JSONL file.",
    )
    parser.add_argument(
        "--vidoseek-page-map",
        type=Path,
        default=DATASETS_ROOT / "vidoseek" / "page_map.jsonl",
        help="ViDoSeek global-page-to-filename mapping.",
    )
    parser.add_argument(
        "--output-name",
        default="layout_ocr",
        help="Output directory name below each dataset (default: layout_ocr).",
    )
    return parser.parse_args()


def require_path(path, kind):
    path = path.expanduser().resolve()
    if kind == "directory" and not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc


def select_text(block):
    for field in ("text", "vlm_text"):
        value = block.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return None, None


def collect_pages(layout_path, page_id):
    pages = defaultdict(list)
    field_counts = defaultdict(int)
    sequence = 0
    for line_number, block in read_jsonl(layout_path):
        try:
            page = page_id(block)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid page index in {layout_path} at line {line_number}"
            ) from exc
        if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
            raise ValueError(
                f"Invalid page index {page!r} in {layout_path} "
                f"at line {line_number}"
            )

        text, field = select_text(block)
        if text is None:
            continue
        layout = block.get("layout")
        if not isinstance(layout, (int, float)) or isinstance(layout, bool):
            layout = sequence
        pages[page].append((layout, sequence, text))
        field_counts[field] += 1
        sequence += 1

    return pages, field_counts


def render_page(blocks):
    blocks.sort(key=lambda item: (item[0], item[1]))
    return "\n\n".join(item[2] for item in blocks)


def write_text_atomic(output_path, text):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_dataset(output_dir, filenames, pages):
    unknown_pages = sorted(set(pages) - set(filenames))
    if unknown_pages:
        preview = ", ".join(map(str, unknown_pages[:10]))
        raise ValueError(
            f"Layout contains pages absent from the target mapping: {preview}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    empty_pages = 0
    for page, filename in sorted(filenames.items()):
        text = render_page(pages.get(page, []))
        if not text:
            empty_pages += 1
        write_text_atomic(output_dir / filename, text)

    expected = set(filenames.values())
    actual = {path.name for path in output_dir.glob("*.txt")}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Output filename mismatch in {output_dir}: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    return len(filenames), empty_pages


def mmd_page_filenames(ppocr_dir):
    filenames = {}
    for path in sorted(ppocr_dir.glob("*.txt")):
        match = PAGE_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected MMD OCR filename: {path}")
        page = int(match.group(1))
        if page in filenames:
            raise ValueError(f"Duplicate MMD page number {page}: {ppocr_dir}")
        filenames[page] = path.name
    if not filenames:
        raise ValueError(f"No page text files found in {ppocr_dir}")
    return filenames


def collect_mmd(data_root, layout_dir, output_name):
    totals = defaultdict(int)
    for dataset_dir in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not dataset_dir.is_dir() or dataset_dir.name == "vidoseek":
            continue
        layout_path = layout_dir / f"{dataset_dir.name}_layout.jsonl"
        if not layout_path.is_file():
            continue
        filenames = mmd_page_filenames(dataset_dir / "ppocr")
        pages, field_counts = collect_pages(
            layout_path, lambda block: block["page_idx"] + 1
        )
        page_count, empty_count = write_dataset(
            dataset_dir / output_name, filenames, pages
        )
        totals["documents"] += 1
        totals["pages"] += page_count
        totals["empty_pages"] += empty_count
        totals["text_blocks"] += field_counts["text"]
        totals["vlm_blocks"] += field_counts["vlm_text"]
    if totals["documents"] == 0:
        raise ValueError(
            f"No data/* directories matched layout files in {layout_dir}"
        )
    return totals


def vidoseek_page_filenames(page_map_path):
    filenames = {}
    for line_number, entry in read_jsonl(page_map_path):
        try:
            page = entry["global_page_id"]
            filename = f"{Path(entry['image']).stem}.txt"
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Invalid ViDoSeek page map at line {line_number}"
            ) from exc
        if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
            raise ValueError(
                f"Invalid global page ID {page!r} at line {line_number}"
            )
        if page in filenames:
            raise ValueError(f"Duplicate ViDoSeek global page ID: {page}")
        filenames[page] = filename

    expected_pages = set(range(1, len(filenames) + 1))
    if set(filenames) != expected_pages:
        raise ValueError("ViDoSeek global page IDs are not contiguous from 1")
    return filenames


def collect_vidoseek(
    data_root, layout_path, page_map_path, output_name
):
    filenames = vidoseek_page_filenames(page_map_path)
    pages, field_counts = collect_pages(
        layout_path, lambda block: block["page_idx"] + 1
    )
    page_count, empty_count = write_dataset(
        data_root / "vidoseek" / output_name, filenames, pages
    )
    return {
        "documents": 1,
        "pages": page_count,
        "empty_pages": empty_count,
        "text_blocks": field_counts["text"],
        "vlm_blocks": field_counts["vlm_text"],
    }


def format_summary(name, totals):
    return (
        f"{name}: {totals['documents']} document(s), "
        f"{totals['pages']} pages, {totals['empty_pages']} empty pages, "
        f"{totals['text_blocks']} text blocks, "
        f"{totals['vlm_blocks']} VLM fallback blocks"
    )


def main():
    args = parse_args()
    data_root = require_path(args.data_root, "directory")
    mmd_layout_dir = require_path(args.mmd_layout_dir, "directory")
    vidoseek_layout = require_path(args.vidoseek_layout, "file")
    vidoseek_page_map = require_path(args.vidoseek_page_map, "file")

    mmd_totals = collect_mmd(data_root, mmd_layout_dir, args.output_name)
    vidoseek_totals = collect_vidoseek(
        data_root,
        vidoseek_layout,
        vidoseek_page_map,
        args.output_name,
    )
    print(format_summary("MMD OCRAG", mmd_totals))
    print(format_summary("ViDoSeek", vidoseek_totals))


if __name__ == "__main__":
    main()
