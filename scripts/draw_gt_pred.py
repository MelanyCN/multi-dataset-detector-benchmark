import argparse
import csv
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


COLOR_GT = (0, 90, 255)
COLOR_PRED = (0, 200, 0)
TEXT_BG_GT = (0, 90, 255, 120)
TEXT_BG_PRED = (0, 200, 0, 120)

try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 12)
    FONT_SMALL = ImageFont.truetype("DejaVuSans.ttf", 11)
except OSError:
    FONT = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser(description="Draw ground truth and RF-DETR predictions.")
    parser.add_argument(
        "--config",
        default="configs/rfdetr_config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def project_root_from_config(config):
    return Path(config.get("project_root", ".")).expanduser().resolve()


def resolve_path(path_value, project_root):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def read_boxes(csv_path, is_pred=False, threshold=0.0):
    data = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            box = {
                "class_name": row["class_name"],
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
            }

            if is_pred:
                box["confidence"] = float(row["confidence"])
                if box["confidence"] < threshold:
                    continue

            data.setdefault(row["image_name"], []).append(box)

    return data


def draw_box_with_label(draw, box, label, color, fill_rgba, img_w, img_h):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))

    if x2 <= x1 or y2 <= y1:
        return

    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    bbox = draw.textbbox((0, 0), label, font=FONT_SMALL)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + 2
    ty = y1 + 2

    if y2 - y1 < th + 6:
        ty = max(0, y1 - th - 4)

    draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=fill_rgba)
    draw.text((tx + 2, ty + 1), label, fill=(255, 255, 255), font=FONT_SMALL)


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = project_root_from_config(config)
    rfdetr_config = config.get("rfdetr", {})
    threshold = float(rfdetr_config.get("confidence_threshold", 0.25))

    dataset_dir = resolve_path(config["output_dataset"], project_root)
    result_dir = resolve_path(config.get("results_dir", "results"), project_root) / "rfdetr"

    img_dir = dataset_dir / "images"
    gt_csv = dataset_dir / "gt_annotations.csv"
    pred_csv = result_dir / "predictions.csv"
    out_dir = result_dir / "images_gt_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_by_img = read_boxes(gt_csv, is_pred=False)
    pred_by_img = read_boxes(pred_csv, is_pred=True, threshold=threshold)
    images = sorted([p for p in img_dir.iterdir() if p.is_file()])

    for idx, img_path in enumerate(images, start=1):
        img_name = img_path.name
        base = Image.open(img_path).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        corner_text = f"RF-DETR | threshold={threshold}"
        cb = draw.textbbox((0, 0), corner_text, font=FONT)
        draw.rectangle([5, 5, cb[2] + 15, cb[3] + 13], fill=(0, 0, 0, 130))
        draw.text((10, 8), corner_text, fill=(255, 255, 255), font=FONT)

        for box in gt_by_img.get(img_name, []):
            draw_box_with_label(
                draw,
                box,
                f"GT: {box['class_name']}",
                COLOR_GT,
                TEXT_BG_GT,
                base.width,
                base.height,
            )

        for box in pred_by_img.get(img_name, []):
            draw_box_with_label(
                draw,
                box,
                f"P_RF: {box['class_name']} {box['confidence']:.2f}",
                COLOR_PRED,
                TEXT_BG_PRED,
                base.width,
                base.height,
            )

        result = Image.alpha_composite(base, overlay).convert("RGB")
        result.save(out_dir / img_name)
        print(f"[{idx}/{len(images)}] {img_name}")

    print(f"\nListo. Imagenes guardadas en: {out_dir}")


if __name__ == "__main__":
    main()
