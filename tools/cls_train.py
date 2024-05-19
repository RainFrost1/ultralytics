from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.yaml')
# print(model)
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="../feature_dataset//caltech-101/", epochs=100, imgsz=224)
