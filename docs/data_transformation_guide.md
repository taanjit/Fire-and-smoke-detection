# Data Transformation Quick Reference

## 🚀 Quick Start

### Run Data Transformation

```bash
cd /home/tadmnit/AI_Team/Anjit/Fire_smoke_detection
source venv/bin/activate
python3 src/fire_smoke_detection/pipeline/stage_02_data_transformation.py
```

### Run Complete Pipeline (Validation + Transformation)

```bash
python3 main.py
```

---

## 📊 What It Does

1. **Resizes images** to 640x640 with letterboxing (maintains aspect ratio)
2. **Adjusts labels** to match transformed image coordinates
3. **Generates YOLO config** (`dataset.yaml`) for training
4. **Creates report** with transformation statistics

---

## 📁 Output Location

```
artifacts/data_transformation/
├── dataset.yaml              # YOLO training config
├── transformation_report.txt # Statistics
├── train/
│   ├── images/              # 877 transformed images
│   └── labels/              # 877 adjusted labels
└── test/
    ├── images/              # 55 transformed images
    └── labels/              # 55 adjusted labels
```

---

## ✅ Verification

Check the transformation report:
```bash
cat artifacts/data_transformation/transformation_report.txt
```

Check YOLO config:
```bash
cat artifacts/data_transformation/dataset.yaml
```

---

## 🎯 Next Steps

Your data is now ready for training! Proceed to:
1. **Model Training** - Train YOLOv8 model
2. **Model Evaluation** - Test on validation set
3. **Deployment** - Create inference pipeline

---

## ⚙️ Configuration

### Change Image Size

Edit `config/config.yaml`:
```yaml
data_transformation:
  image_size: [640, 640]  # Change to [416, 416] or [1280, 1280]
```

### Augmentation Settings

Edit `params.yaml`:
```yaml
AUGMENTATION:
  fliplr: 0.5      # Horizontal flip probability
  mosaic: 1.0      # Mosaic augmentation
  hsv_h: 0.015     # Hue variation
  # ... more settings
```

> **Note**: Augmentation is applied during training, not transformation.
