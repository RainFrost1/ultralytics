import onnx
import tensorflow as tf

# 加载ONNX模型
onnx_model = onnx.load('best.onnx')

# 创建TFLite Converter
converter = tf.lite.TFLiteConverter.from_onnx_model(onnx_model)

# 转换模型
tflite_model = converter.convert()

# 保存TFLite模型
with open('yolo11_feature.tflite', 'wb') as f:
    f.write(tflite_model)

