from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-feature-finetune.yaml')
model.load_state_dict('xxxxx')
# print(model)
#model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/data/feature_data//shitu_dataset/", cfg="./tools/default_finetune.yaml")
model.export(format='onnx', dynamic=True, imgsz=224)
model.export(format='tflite', int8=False, batch=1, half=True, imgsz=224)
