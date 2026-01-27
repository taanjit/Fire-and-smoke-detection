# Pipeline Quick Reference

## 🚀 Quick Start

### Run Complete Pipeline

```bash
cd /home/tadmnit/AI_Team/Anjit/Fire_smoke_detection
source venv/bin/activate
python3 run_pipeline.py
```

---

## 📋 Common Commands

### Full Pipeline
```bash
python3 run_pipeline.py
```

### Skip Validation
```bash
python3 run_pipeline.py --skip validation
```

### Start from Training
```bash
python3 run_pipeline.py --start-from training
```

### Skip Multiple Stages
```bash
python3 run_pipeline.py --skip validation transformation
```

---

## 🎯 Individual Stages

```bash
# Validation only
python3 src/fire_smoke_detection/pipeline/stage_01_data_validation.py

# Transformation only
python3 src/fire_smoke_detection/pipeline/stage_02_data_transformation.py

# Training only
python3 train.py

# Evaluation only
python3 evaluate.py
```

---

## 📊 Pipeline Stages

1. **Data Validation** - Verify dataset integrity
2. **Data Transformation** - Preprocess images and labels
3. **Model Training** - Train YOLOv8 model
4. **Model Evaluation** - Calculate metrics and visualizations

---

## 📁 Output Locations

```
artifacts/
├── data_validation/status.txt
├── data_transformation/dataset.yaml
├── model_training/best.pt
├── model_evaluation/metrics.json
└── pipeline_report.txt
```

---

## ⏱️ Typical Execution Times

**YOLOv8n on 877 images:**
- Validation: ~2 seconds
- Transformation: ~3 seconds
- Training (100 epochs): ~10-15 minutes (GPU) / 2-4 hours (CPU)
- Evaluation: ~5 seconds

**Total**: ~15-20 minutes (GPU) / 2-4 hours (CPU)

---

## 🔧 Configuration

### Change Model Size
Edit `params.yaml`:
```yaml
MODEL:
  variant: n  # Options: n, s, m, l, x
```

### Adjust Training Duration
```yaml
TRAINING:
  epochs: 100
  batch_size: 16
```

---

## 💡 Tips

- Use `--skip validation` for faster iteration
- Use `--start-from training` to retrain without data prep
- Check `artifacts/pipeline_report.txt` for execution summary
- Monitor training with TensorBoard (optional)

---

## 🐛 Troubleshooting

### Pipeline Fails
1. Check error message in console
2. Review `artifacts/logs/` (if logging enabled)
3. Check stage-specific reports in `artifacts/`
4. Resume from failed stage:
   ```bash
   python3 run_pipeline.py --start-from <stage>
   ```

### Out of Memory
Reduce batch size in `params.yaml`:
```yaml
TRAINING:
  batch_size: 4
```

---

## 📚 Documentation

- Full Guide: `docs/pipeline_guide.md`
- Training Guide: `docs/training_guide.md`
- Utilities Guide: `docs/utilities_guide.md`
