from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n-feature.yaml')
# print(model)
#model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
model.load('./yolo11n.pt')
results = model.train(data="/data/ygy/data/food+general/", cfg="./tools/default_food_total_epoch200_feature1024.yaml")
model.export(format='onnx', dynamic=True, imgsz=224)
model.export(format='tflite', int8=False, batch=1, half=True, imgsz=224)
