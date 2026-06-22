from ultralytics import YOLO

# Paths - update these to your setup
PRETRAINED_MODEL = r'D:\Codes\fdss\finetuned_training_runs\chesspiece_finetune\weights\best.pt'
DATASET_YAML = r'D:\Codes\fdss\Combined_data\data.yaml'  # New combined dataset


# Load the pretrained model
model = YOLO(PRETRAINED_MODEL)

# Fine-tune with aggresssive default augmentations (augment=True)
results = model.train(
    data=DATASET_YAML,
    epochs=100,
    batch=8,
    imgsz=640,
    augment=True,         # Enable default augmentations
    patience=10,          # Early stopping patience on validation loss
    name='chesspiece_finetuned_aggaug',
    project='finetuned_training_runs',
    exist_ok=True,
    save_period=1         # Save checkpoint every epoch
)

print("✅ Fine-tuning completed! Best weights saved at:")
print("finetuned_training_runs/chesspiece_finetuned_aggaug/weights/best.pt")
