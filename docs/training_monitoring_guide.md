# Training and Evaluation Guide - Fire and Smoke Detection

## 🎯 Current Training Status

Your model is currently training! Here's what the output means:

### Understanding Training Output

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
17/100      2.55G     0.9791      1.257      1.468         38        640
```

**Metrics Explained:**
- **Epoch**: Current epoch out of total (17/100)
- **GPU_mem**: GPU memory usage (2.55G)
- **box_loss**: Bounding box localization loss (lower is better)
- **cls_loss**: Classification loss (lower is better)
- **dfl_loss**: Distribution focal loss (lower is better)
- **Instances**: Number of objects in current batch
- **Size**: Input image size (640x640)

### Validation Metrics

```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
all         55         57      0.673      0.749      0.734      0.386
```

**Metrics Explained:**
- **Images**: Number of validation images (55)
- **Instances**: Total objects in validation set (57)
- **Box(P)**: Precision = 0.673 (67.3% of detections are correct)
- **R**: Recall = 0.749 (74.9% of objects are detected)
- **mAP50**: Mean Average Precision at IoU=0.5 = 0.734 (73.4%)
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95 = 0.386 (38.6%)

---

## 📊 Monitoring Training Progress

### What to Watch

1. **Loss Values** (should decrease over time):
   - box_loss: ~1.0 → should go down to ~0.5-0.7
   - cls_loss: ~1.3 → should go down to ~0.5-0.8
   - dfl_loss: ~1.5 → should go down to ~0.8-1.2

2. **mAP50** (should increase over time):
   - Current: 0.734 (73.4%)
   - Good target: > 0.80 (80%)
   - Excellent: > 0.90 (90%)

3. **Precision and Recall** (should increase):
   - Current: P=0.673, R=0.749
   - Good target: Both > 0.75
   - Excellent: Both > 0.85

### Training Progress Indicators

✅ **Good Signs:**
- Loss values decreasing
- mAP50 increasing
- Precision and recall improving
- Training speed stable (~30 it/s)

⚠️ **Warning Signs:**
- Loss values increasing
- mAP50 decreasing
- Training speed dropping significantly
- GPU memory errors

---

## ⏱️ Estimated Training Time

**Current Progress:**
- Epoch: 18/100
- Remaining: 82 epochs
- Speed: ~1.8s per epoch

**Estimated Time Remaining:**
- 82 epochs × 1.8s = ~148 seconds = **~2.5 minutes**

**Total Training Time:**
- 100 epochs × 1.8s = ~180 seconds = **~3 minutes**

> **Note**: This is very fast because you're using GPU and have a small dataset (877 images)

---

## 📁 Training Outputs

### During Training

Files being created in `artifacts/model_training/train/`:

```
train/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Latest model (last epoch)
├── results.csv          # Metrics per epoch
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1 score curve
├── PR_curve.png         # Precision-Recall curve
├── P_curve.png          # Precision curve
└── R_curve.png          # Recall curve
```

### After Training

Best model copied to:
```
artifacts/model_training/best.pt
```

---

## 🎓 Understanding Your Results

### Current Performance (Epoch 17)

| Metric | Value | Rating |
|--------|-------|--------|
| mAP@0.5 | 0.734 | 🟢 Good |
| mAP@0.5:0.95 | 0.386 | 🟡 Fair |
| Precision | 0.673 | 🟡 Fair |
| Recall | 0.749 | 🟢 Good |

**Interpretation:**
- **mAP@0.5 = 73.4%**: Your model correctly detects fire/smoke 73.4% of the time (good!)
- **Precision = 67.3%**: When model says "fire/smoke", it's correct 67.3% of the time
- **Recall = 74.9%**: Model finds 74.9% of all fire/smoke instances
- **mAP@0.5:0.95 = 38.6%**: Bounding boxes could be more accurate

### What This Means

✅ **Strengths:**
- Good at detecting fire and smoke (high recall)
- Decent overall accuracy (mAP@0.5 > 70%)

⚠️ **Areas for Improvement:**
- Some false positives (precision could be higher)
- Bounding box accuracy (mAP@0.5:0.95 is lower)

---

## 🔍 After Training Completes

### Step 1: Check Training Results

```bash
# View training curves
xdg-open artifacts/model_training/train/results.png

# View confusion matrix
xdg-open artifacts/model_training/train/confusion_matrix.png

# Check metrics CSV
cat artifacts/model_training/train/results.csv
```

### Step 2: Review Best Model

```bash
# Check best model file
ls -lh artifacts/model_training/best.pt

# View model info
python3 -c "from ultralytics import YOLO; model = YOLO('artifacts/model_training/best.pt'); model.info()"
```

### Step 3: Evaluate Model

The pipeline will automatically run evaluation after training completes. You'll see:

```
>>>>>> Model Evaluation Stage Started <<<<<<
Loading model from: artifacts/model_training/best.pt
Evaluating on test data...
...
>>>>>> Model Evaluation Stage Completed Successfully <<<<<<
mAP@0.5: 0.7234
Precision: 0.7156
Recall: 0.7389
F1-Score: 0.7271
```

### Step 4: Check Evaluation Results

```bash
# View metrics
cat artifacts/model_evaluation/metrics.json

# View evaluation report
cat artifacts/model_evaluation/evaluation_report.txt

# View sample predictions
ls artifacts/model_evaluation/predictions/
```

---

## 🚀 Improving Model Performance

### If Performance is Not Satisfactory

#### 1. Train Longer

Edit `params.yaml`:
```yaml
TRAINING:
  epochs: 200  # Increase from 100
```

#### 2. Use Larger Model

Edit `params.yaml`:
```yaml
MODEL:
  variant: s  # Change from 'n' to 's' (small)
  # Options: n (nano), s (small), m (medium), l (large), x (extra-large)
```

#### 3. Increase Data Augmentation

Edit `params.yaml`:
```yaml
AUGMENTATION:
  fliplr: 0.7      # Increase horizontal flip
  mosaic: 1.0      # Keep mosaic augmentation
  mixup: 0.1       # Add mixup augmentation
  hsv_h: 0.02      # Increase hue variation
  hsv_s: 0.8       # Increase saturation
  hsv_v: 0.5       # Increase value
```

#### 4. Adjust Learning Rate

Edit `params.yaml`:
```yaml
TRAINING:
  learning_rate: 0.001  # Reduce from 0.01 for fine-tuning
```

#### 5. Collect More Data

- Add more training images
- Ensure balanced classes (equal fire and smoke examples)
- Include diverse scenarios (day/night, indoor/outdoor, etc.)

---

## 📊 Expected Final Results

### Good Model Performance

For fire and smoke detection:

| Metric | Target | Excellent |
|--------|--------|-----------|
| mAP@0.5 | > 0.75 | > 0.85 |
| mAP@0.5:0.95 | > 0.45 | > 0.60 |
| Precision | > 0.75 | > 0.85 |
| Recall | > 0.75 | > 0.85 |
| F1-Score | > 0.75 | > 0.85 |

### Your Current Trajectory

Based on epoch 17 results, you're on track for:
- **mAP@0.5**: ~0.75-0.80 (Good to Very Good)
- **Precision**: ~0.70-0.75 (Fair to Good)
- **Recall**: ~0.75-0.80 (Good to Very Good)

---

## 🎯 Next Steps After Training

### 1. Review Complete Pipeline Report

```bash
cat artifacts/pipeline_report.txt
```

### 2. Test Model on New Images

```python
from ultralytics import YOLO

# Load best model
model = YOLO('artifacts/model_training/best.pt')

# Run prediction
results = model.predict(
    source='path/to/test/image.jpg',
    conf=0.25,
    save=True
)
```

### 3. Deploy Model

Options:
- Create inference script
- Build web application
- Deploy to edge device
- Create REST API

### 4. Monitor in Production

- Track detection accuracy
- Log false positives/negatives
- Collect edge cases for retraining

---

## 🔧 Troubleshooting

### Training is Slow

**If using CPU:**
- Expected: 2-4 hours for 100 epochs
- Solution: Use GPU or reduce epochs

**If using GPU but slow:**
- Check GPU utilization: `nvidia-smi`
- Reduce batch size if memory limited
- Close other GPU applications

### Loss Not Decreasing

**Possible causes:**
- Learning rate too high
- Model too small
- Data quality issues

**Solutions:**
- Reduce learning rate to 0.001
- Use larger model variant
- Check data labels

### mAP Not Improving

**Possible causes:**
- Insufficient training
- Poor data quality
- Class imbalance

**Solutions:**
- Train for more epochs
- Verify label accuracy
- Balance dataset classes

---

## 📈 Monitoring Tools

### TensorBoard (Optional)

If enabled in `params.yaml`:

```bash
tensorboard --logdir artifacts/model_training/train
```

Then open: http://localhost:6006

### Real-time Monitoring

Watch training progress:

```bash
watch -n 1 tail -20 artifacts/model_training/train/results.csv
```

---

## ✅ Success Criteria

Your model is ready for deployment when:

- ✅ mAP@0.5 > 0.75
- ✅ Precision > 0.70
- ✅ Recall > 0.70
- ✅ F1-Score > 0.70
- ✅ Confusion matrix shows good performance on all classes
- ✅ Sample predictions look accurate

---

## 🎉 What to Expect

Based on your current progress (epoch 17/100):

**Final Expected Metrics:**
- mAP@0.5: **0.75-0.80** (Good)
- Precision: **0.70-0.75** (Fair to Good)
- Recall: **0.75-0.80** (Good)
- F1-Score: **0.72-0.77** (Good)

**Training will complete in ~2-3 minutes!**

After completion, you'll have:
- ✅ Trained model weights
- ✅ Training curves and plots
- ✅ Evaluation metrics
- ✅ Sample predictions
- ✅ Complete pipeline report

---

## 📚 Additional Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com
- **Training Tips**: `docs/training_guide.md`
- **Pipeline Guide**: `docs/pipeline_guide.md`
- **Utilities Guide**: `docs/utilities_guide.md`

---

**Your training is progressing well! Just wait for it to complete (~2-3 minutes remaining).**
