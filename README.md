# Multi-Dataset Detector Benchmark

Pipeline for building a balanced object-detection benchmark dataset from multiple sources, running pretrained detector inference, and saving metrics, predictions, and GT-vs-prediction visualizations.

## Repository Layout

```text
multi-dataset-detector-benchmark/
├── configs/
│   └── rfdetr_config.yaml
├── docker/
│   ├── rfdetr/Dockerfile
│   ├── dfine/Dockerfile
│   └── rtdetr/Dockerfile
├── scripts/
│   ├── create_test_dataset.py
│   ├── run_rfdetr_benchmark.py
│   ├── draw_gt_pred.py
│   └── check_gt_coverage.py
├── data/
│   ├── raw/
│   └── benchmark_test/
├── results/
├── docs/
│   └── run_report.md
└── README.md
```

## Data Policy

Do not commit real datasets, full result folders, downloaded model weights, exported engines, or compressed artifacts. Keep small illustrative files in `examples/` only when needed.

Ignored local artifacts include:

```text
datasets/
data/raw/
data/benchmark_test*/
results/
articulos_dataset/
*.pth
*.pt
*.onnx
*.engine
*.zip
```

## Configuration

Edit `configs/rfdetr_config.yaml` if your datasets live somewhere else. Paths can be absolute or relative to `project_root`.

```yaml
project_root: /workspace/project
output_dataset: data/benchmark_test_300_v2
num_images_per_dataset: 100
seed: 42
results_dir: results
```

## Docker

Build the RF-DETR image:

```bash
docker build -t rfdetr-benchmark -f docker/rfdetr/Dockerfile .
```

Run it with local data and result mounts:

```bash
docker run --gpus all -it \
  -v "$(pwd)":/workspace/project \
  -v /ruta/local/datasets:/workspace/project/data/raw \
  -v /ruta/local/results:/workspace/project/results \
  rfdetr-benchmark
```

Inside the container:

```bash
cd /workspace/project
python scripts/create_test_dataset.py --config configs/rfdetr_config.yaml
python scripts/check_gt_coverage.py --config configs/rfdetr_config.yaml
python scripts/run_rfdetr_benchmark.py --config configs/rfdetr_config.yaml
python scripts/draw_gt_pred.py --config configs/rfdetr_config.yaml
```

## Expected Dataset Sources

The default config expects:

```text
data/raw/AAU_RainSnow_Traffic_Surveillance/aauRainSnow-rgb.json
data/raw/Adverse_Weather/train/_annotations.coco.json
data/raw/Adverse_Weather/valid/_annotations.coco.json
data/raw/Adverse_Weather/test/_annotations.coco.json
data/raw/DAWN/
```

DAWN is expected in YOLO Darknet format with image labels in `.txt` files and, ideally, a `.names` file for class names.
