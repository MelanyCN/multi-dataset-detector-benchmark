# Run Report

## Objective

This project creates a balanced benchmark dataset from multiple object-detection datasets, runs pretrained detector inference, and stores metrics, predictions, and GT-vs-prediction visualizations.

## Current RF-DETR Flow

1. Create the benchmark subset:

   ```bash
   python scripts/create_test_dataset.py --config configs/rfdetr_config.yaml
   ```

2. Check that every selected image has ground-truth boxes:

   ```bash
   python scripts/check_gt_coverage.py --config configs/rfdetr_config.yaml
   ```

3. Run RF-DETR inference:

   ```bash
   python scripts/run_rfdetr_benchmark.py --config configs/rfdetr_config.yaml
   ```

4. Draw GT and predictions:

   ```bash
   python scripts/draw_gt_pred.py --config configs/rfdetr_config.yaml
   ```

## Outputs

- `data/benchmark_test_300_v2/image_source_map.csv`
- `data/benchmark_test_300_v2/gt_annotations.csv`
- `results/rfdetr/predictions.csv`
- `results/rfdetr/image_metrics.csv`
- `results/rfdetr/warmup_metrics.csv`
- `results/rfdetr/summary_metrics.json`
- `results/rfdetr/images_bbox/`
- `results/rfdetr/images_gt_pred/`

These outputs are local artifacts and are ignored by Git.
