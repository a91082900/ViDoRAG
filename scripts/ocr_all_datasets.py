#!/usr/bin/env python3
"""Run one shared PP-OCRv3 pipeline over every dataset below data/."""

import argparse
import multiprocessing
import sys
from pathlib import Path

try:
    from .ocr_triditional import (
        PROJECT_ROOT,
        SUPPORTED_IMAGE_SUFFIXES,
        build_ocr_pipeline,
        decode_image,
        load_dependencies,
        resolve_model_files,
        to_layout_preserving_text,
        write_text_atomic,
    )
except ImportError:
    from ocr_triditional import (
        PROJECT_ROOT,
        SUPPORTED_IMAGE_SUFFIXES,
        build_ocr_pipeline,
        decode_image,
        load_dependencies,
        resolve_model_files,
        to_layout_preserving_text,
        write_text_atomic,
    )


_WORKER_PIPELINE = None
_WORKER_CV2 = None
_WORKER_MIN_SCORE = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract OCR text from every data/*/img directory."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing dataset subdirectories (default: data/).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory containing PP-OCR models and en_dict.txt (default: models/).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="FastDeploy inference device (default: cpu).",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID; ignored for CPU inference (default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of OCR worker processes (default: 4).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=6,
        help="CPU threads used by each worker's FastDeploy runtime (default: 6).",
    )
    parser.add_argument(
        "--mkldnn-cache-size",
        type=int,
        default=1,
        help="Maximum cached MKL-DNN dynamic input shapes per runtime (default: 1).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Discard recognized text below this confidence (default: 0.6).",
    )
    args = parser.parse_args()
    if not 0 <= args.min_score <= 1:
        parser.error("--min-score must be between 0 and 1")
    if args.workers <= 0:
        parser.error("--workers must be greater than 0")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be greater than 0")
    if args.mkldnn_cache_size <= 0:
        parser.error("--mkldnn-cache-size must be greater than 0")
    if args.device == "gpu" and args.workers != 1:
        parser.error("--device gpu requires --workers 1")
    return args


def find_datasets(data_root):
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    datasets = []
    for dataset_dir in sorted(data_root.iterdir(), key=lambda path: path.name):
        image_dir = dataset_dir / "img"
        if not image_dir.is_dir():
            continue
        if any(
            path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            for path in image_dir.iterdir()
        ):
            datasets.append(dataset_dir)
    return datasets


def collect_tasks(datasets):
    tasks = []
    skipped = 0
    for dataset_dir in datasets:
        output_dir = dataset_dir / "ppocr"
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted((dataset_dir / "img").iterdir()):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
            ):
                continue
            output_path = output_dir / f"{image_path.stem}.txt"
            if output_path.exists():
                skipped += 1
            else:
                tasks.append((str(image_path), str(output_path)))
    return tasks, skipped


def initialize_worker(
    model_files,
    device,
    device_id,
    cpu_threads,
    mkldnn_cache_size,
    min_score,
):
    global _WORKER_PIPELINE, _WORKER_CV2, _WORKER_MIN_SCORE
    fd, cv2, _ = load_dependencies()
    _WORKER_PIPELINE = build_ocr_pipeline(
        fd,
        model_files,
        device,
        device_id,
        cpu_threads,
        mkldnn_cache_size,
    )
    _WORKER_CV2 = cv2
    _WORKER_MIN_SCORE = min_score


def process_image_task(task):
    image_path = Path(task[0])
    output_path = Path(task[1])
    if output_path.exists():
        return "skipped", str(image_path), None

    try:
        image = decode_image(image_path, _WORKER_CV2)
        result = _WORKER_PIPELINE.predict(image)
        text = to_layout_preserving_text(result, _WORKER_MIN_SCORE)
        write_text_atomic(output_path, text)
        return "processed", str(image_path), None
    except Exception as exc:
        return "failed", str(image_path), f"{type(exc).__name__}: {exc}"


def run(args):
    datasets = find_datasets(args.data_root)
    if not datasets:
        print(f"No datasets with supported images found below {args.data_root}")
        return 0

    tasks, total_skipped = collect_tasks(datasets)
    if not tasks:
        print(
            f"All datasets completed: 0 processed, {total_skipped} skipped, "
            "0 failed."
        )
        return 0

    model_files = resolve_model_files(args.model_root)
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Missing OCR dependency. Create a Python 3.10 environment and run "
            "`python -m pip install -r requirements-ocr.txt`."
        ) from exc

    worker_count = min(args.workers, len(tasks))
    print(
        f"Found {len(datasets)} datasets: {len(tasks)} queued, "
        f"{total_skipped} already completed. Starting {worker_count} workers "
        f"with {args.cpu_threads} CPU threads each."
    )
    total_processed = 0
    total_failures = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=initialize_worker,
        initargs=(
            model_files,
            args.device,
            args.device_id,
            args.cpu_threads,
            args.mkldnn_cache_size,
            args.min_score,
        ),
    ) as pool:
        results = pool.imap_unordered(process_image_task, tasks, chunksize=1)
        for status, image_path, error in tqdm(
            results,
            total=len(tasks),
            desc="OCR all datasets",
        ):
            if status == "processed":
                total_processed += 1
            elif status == "skipped":
                total_skipped += 1
            else:
                total_failures.append((image_path, error))
                tqdm.write(f"Failed to process {image_path}: {error}")

    print(
        f"All datasets completed: {total_processed} processed, "
        f"{total_skipped} skipped, {len(total_failures)} failed."
    )
    return 1 if total_failures else 0


def main():
    args = parse_args()
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed outputs are preserved for the next run.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
