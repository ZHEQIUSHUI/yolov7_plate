import onnx

input_path = "yolov7-lite-s.onnx"
output_path = "yolov7-lite-s-ncnn.onnx"
input_names = ["images"]
output_names = ["onnx::Slice_994","onnx::Slice_1183","onnx::Slice_1372"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
