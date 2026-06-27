#!/usr/bin/env python3
"""Run PP-OCRv3 over a dataset's images with FastDeploy."""

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract English text from data/<dataset>/img with PP-OCRv3."
    )
    parser.add_argument(
        "--dataset",
        default="ExampleDataset",
        help="Dataset directory name below data/ (default: ExampleDataset).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT,
        help=(
            "Directory containing the three PP-OCR model directories and "
            "en_dict.txt (default: repository root)."
        ),
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
        "--cpu-threads",
        type=int,
        default=6,
        help="CPU threads used by each FastDeploy runtime (default: 6).",
    )
    parser.add_argument(
        "--mkldnn-cache-size",
        type=int,
        default=1,
        help="Maximum cached MKL-DNN dynamic input shapes (default: 1).",
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
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be greater than 0")
    if args.mkldnn_cache_size <= 0:
        parser.error("--mkldnn-cache-size must be greater than 0")
    return args


def load_dependencies():
    try:
        import cv2
        import fastdeploy as fd
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Missing OCR dependency. Create a Python 3.10 environment and run "
            "`python -m pip install -r requirements-ocr.txt`."
        ) from exc
    return fd, cv2, tqdm


def require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return str(path)


def resolve_model_files(model_root):
    model_root = model_root.expanduser().resolve()
    return {
        "det_model": require_file(
            model_root / "en_PP-OCRv3_det_infer" / "inference.pdmodel",
            "detection model",
        ),
        "det_params": require_file(
            model_root / "en_PP-OCRv3_det_infer" / "inference.pdiparams",
            "detection parameters",
        ),
        "cls_model": require_file(
            model_root / "ch_ppocr_mobile_v2.0_cls_infer" / "inference.pdmodel",
            "classification model",
        ),
        "cls_params": require_file(
            model_root / "ch_ppocr_mobile_v2.0_cls_infer" / "inference.pdiparams",
            "classification parameters",
        ),
        "rec_model": require_file(
            model_root / "en_PP-OCRv3_rec_infer" / "inference.pdmodel",
            "recognition model",
        ),
        "rec_params": require_file(
            model_root / "en_PP-OCRv3_rec_infer" / "inference.pdiparams",
            "recognition parameters",
        ),
        "rec_labels": require_file(model_root / "en_dict.txt", "English dictionary"),
    }


def build_runtime_option(
    fd,
    device,
    device_id,
    cpu_threads=None,
    mkldnn_cache_size=1,
):
    option = fd.RuntimeOption()
    if device == "gpu":
        option.use_gpu(device_id)
    else:
        option.use_cpu()
        if cpu_threads is not None:
            option.set_cpu_thread_num(cpu_threads)
        option.paddle_infer_option.mkldnn_cache_size = mkldnn_cache_size
    return option


def build_ocr_pipeline(
    fd,
    model_files,
    device,
    device_id,
    cpu_threads=None,
    mkldnn_cache_size=1,
):
    det_option = build_runtime_option(
        fd, device, device_id, cpu_threads, mkldnn_cache_size
    )
    cls_option = build_runtime_option(
        fd, device, device_id, cpu_threads, mkldnn_cache_size
    )
    rec_option = build_runtime_option(
        fd, device, device_id, cpu_threads, mkldnn_cache_size
    )

    det_model = fd.vision.ocr.DBDetector(
        model_files["det_model"],
        model_files["det_params"],
        runtime_option=det_option,
    )
    cls_model = fd.vision.ocr.Classifier(
        model_files["cls_model"],
        model_files["cls_params"],
        runtime_option=cls_option,
    )
    rec_model = fd.vision.ocr.Recognizer(
        model_files["rec_model"],
        model_files["rec_params"],
        model_files["rec_labels"],
        runtime_option=rec_option,
    )

    det_model.preprocessor.max_side_len = 960
    det_model.postprocessor.det_db_thresh = 0.3
    det_model.postprocessor.det_db_box_thresh = 0.6
    det_model.postprocessor.det_db_unclip_ratio = 1.5
    det_model.postprocessor.det_db_score_mode = "slow"
    det_model.postprocessor.use_dilation = False
    cls_model.postprocessor.cls_thresh = 0.9

    pipeline = fd.vision.ocr.PPOCRv3(
        det_model=det_model,
        cls_model=cls_model,
        rec_model=rec_model,
    )
    # FastDeploy's native PPOCRv3 pipeline keeps pointers to these objects but
    # does not retain their Python wrappers. Keep explicit references alive for
    # as long as the pipeline is used.
    pipeline._fastdeploy_refs = (
        det_model,
        cls_model,
        rec_model,
        det_option,
        cls_option,
        rec_option,
    )
    return pipeline


def decode_image(image_path, cv2):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    return image


def calculate_spaces_and_newlines(
    current_box, previous_box, space_threshold=45, line_threshold=15
):
    if abs(current_box[1] - previous_box[1]) < line_threshold:
        spaces = max(1, int(abs(current_box[0] - previous_box[0]) / space_threshold))
        return spaces, 0
    newlines = max(1, int(abs(current_box[1] - previous_box[1]) / line_threshold))
    return 0, newlines


def to_layout_preserving_text(result, min_score):
    text_boxes = []
    for box, text, score in zip(result.boxes, result.text, result.rec_scores):
        if score < min_score:
            continue
        coords = [(box[index], box[index + 1]) for index in range(0, len(box), 2)]
        center_x = (coords[0][0] + coords[2][0]) / 2
        center_y = (coords[0][1] + coords[2][1]) / 2
        text_boxes.append((center_x, center_y, text))

    text_boxes.sort(key=lambda item: (item[1], item[0]))
    merged_text = []
    previous_box = None
    for box in text_boxes:
        if previous_box is not None:
            spaces, newlines = calculate_spaces_and_newlines(box, previous_box)
            merged_text.append("\n" * newlines + " " * spaces)
        merged_text.append(box[2])
        previous_box = box
    return "".join(merged_text)


def write_text_atomic(output_path, text):
    output_path = Path(output_path)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_dataset(dataset_dir, ocr_pipeline, cv2, tqdm, min_score):
    dataset_dir = Path(dataset_dir)
    input_dir = dataset_dir / "img"
    output_dir = dataset_dir / "ppocr"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Dataset image directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return 0, 0, []

    processed = 0
    skipped = 0
    failures = []
    for image_path in tqdm(image_files, desc=f"OCR {dataset_dir.name}"):
        output_path = output_dir / f"{image_path.stem}.txt"
        if output_path.exists():
            skipped += 1
            continue
        try:
            image = decode_image(image_path, cv2)
            result = ocr_pipeline.predict(image)
            text = to_layout_preserving_text(result, min_score)
            write_text_atomic(output_path, text)
            processed += 1
        except Exception as exc:
            failures.append((image_path, exc))
            tqdm.write(f"Failed to process {image_path}: {exc}")

    print(
        f"Completed {dataset_dir.name}: {processed} processed, {skipped} skipped, "
        f"{len(failures)} failed. Output: {output_dir}"
    )
    return processed, skipped, failures


def run(args):
    dataset_dir = PROJECT_ROOT / "data" / args.dataset
    model_files = resolve_model_files(args.model_root)
    fd, cv2, tqdm = load_dependencies()
    ocr_pipeline = build_ocr_pipeline(
        fd,
        model_files,
        args.device,
        args.device_id,
        args.cpu_threads,
        args.mkldnn_cache_size,
    )
    _, _, failures = process_dataset(
        dataset_dir,
        ocr_pipeline,
        cv2,
        tqdm,
        args.min_score,
    )
    return 1 if failures else 0


def main():
    args = parse_args()
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
