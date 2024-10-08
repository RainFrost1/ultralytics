from ultralytics import YOLO

# Load a model
model = YOLO('./runs/feature/train11/weights/last.pt')
# print(model)
#model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.val(data="../feature_dataset//caltech-101/", cfg="./tools/default.yaml")
#model.export(format='onnx', dynamic=True, imgsz=224)
#model.export(format='tflite', int8=False, batch=1, half=True, imgsz=224)
