# Model Training & Evaluation - Quick Reference

## 🚀 Quick Start

### Train Model

```bash
cd /home/tadmnit/AI_Team/Anjit/Fire_smoke_detection
source venv/bin/activate
python3 train.py
```

### Evaluate Model

```bash
python3 evaluate.py
```

### Run Complete Pipeline

```bash
python3 main.py  # Validation → Transformation → Training → Evaluation
```

---

## ⚙️ Configuration

### Change Model Size

Edit `params.yaml`:
```yaml
MODEL:
  variant: n  # Options: n, s, m, l, x
```

**Variants:**
- `n` (nano): Fastest, smallest
- `s` (small): Good balance
- `m` (medium): Better accuracy
- `l` (large): High accuracy
- `x` (extra-large): Best accuracy

### Adjust Training Duration

```yaml
TRAINING:
  epochs: 100      # Number of training epochs
  batch_size: 16   # Reduce if GPU memory limited
  patience: 50     # Early stopping patience
```

### Modify Augmentation

```yaml
AUGMENTATION:
  fliplr: 0.5      # Horizontal flip probability
  mosaic: 1.0      # Mosaic augmentation
  hsv_h: 0.015     # Hue variation
```

---

## 📊 Understanding Metrics

### mAP (Mean Average Precision)
- **mAP@0.5**: Precision at 50% overlap threshold
- **mAP@0.5:0.95**: Average across multiple thresholds
- **Range**: 0-1 (higher is better)
- **Good**: > 0.70

### Precision
- How many detections are correct
- `TP / (TP + FP)`
- **Good**: > 0.70

### Recall
- How many objects were found
- `TP / (TP + FN)`
- **Good**: > 0.70

### F1-Score
- Balance of precision and recall
- `2 × (P × R) / (P + R)`
- **Good**: > 0.70

---

## 📁 Output Files

### Training Outputs

```
artifacts/model_training/
├── best.pt                    # Best model weights
├── model_config.yaml          # Training config
└── train/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv            # Metrics per epoch
    ├── results.png            # Training curves
    └── confusion_matrix.png
```

### Evaluation Outputs

```
artifacts/model_evaluation/
├── metrics.json               # Detailed metrics
├── evaluation_report.txt      # Human-readable report
├── evaluation_metrics.png     # Metrics visualization
└── predictions/               # Sample predictions
```

---

## ⏱️ Training Time

**YOLOv8n on 877 images:**
- **GPU**: ~10-15 minutes (100 epochs)
- **CPU**: ~2-4 hours (100 epochs)

---

## 🎯 Expected Results

**Good Model:**
- mAP@0.5: 0.70 - 0.90
- Precision: 0.70 - 0.90
- Recall: 0.70 - 0.90
- F1-Score: 0.70 - 0.90

---

## 🔧 Troubleshooting

### GPU Not Working

Force CPU in `params.yaml`:
```yaml
TRAINING:
  device: cpu
```

### Out of Memory

Reduce batch size:
```yaml
TRAINING:
  batch_size: 4  # or even 1
```

### Poor Performance

1. Train longer (more epochs)
2. Use larger model variant
3. Increase augmentation
4. Check data quality

---

## 📈 Monitor Training

### Console Output

```
Epoch 1/100: 100%|████| 55/55 [00:15<00:00]
  mAP50: 0.698  Precision: 0.654  Recall: 0.712
```

### TensorBoard (Optional)

```bash
tensorboard --logdir artifacts/model_training/train
```

---

## 🎓 Next Steps

1. **Review Results**: Check metrics and confusion matrix
2. **Improve**: Adjust hyperparameters if needed
3. **Deploy**: Create inference pipeline
4. **Production**: Build web app and containerize

---

## 💡 Tips

- Start with `variant: n` for quick testing
- Use GPU for faster training
- Monitor validation metrics to prevent overfitting
- Save best model automatically enabled
- Early stopping prevents wasted time

---

## 📚 Files Created

- `train.py` - Standalone training script
- `evaluate.py` - Standalone evaluation script
- `src/fire_smoke_detection/components/model_training.py`
- `src/fire_smoke_detection/components/model_evaluation.py`
- `src/fire_smoke_detection/pipeline/stage_03_model_training.py`
- `src/fire_smoke_detection/pipeline/stage_04_model_evaluation.py`
