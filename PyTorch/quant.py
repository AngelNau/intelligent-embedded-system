import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, CalibrationMethod, QuantFormat

model_name = "tinyCnn"
input_name = "input"
input_size = 128
batch_size = 128
num_calib_batches = 50

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.PILToTensor(),
])

calib_dataset = torchvision.datasets.ImageFolder("../flower_photos/", transform=transform)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

class CalibrationReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name, max_batches=None, enforce_batch1=False):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.input_name = input_name
        self.max_batches = max_batches
        self.batches_yielded = 0
        self.enforce_batch1 = enforce_batch1

    def get_next(self):
        # Stop condition must return None (not an empty dict)
        if self.max_batches is not None and self.batches_yielded >= self.max_batches:
            return {}
        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return {}

        # Torch tensors are NCHW; convert to numpy float32
        # batch = images.numpy().astype(np.float32)
        batch = images.numpy().astype(np.uint8)

        # If the ONNX model input is [1,C,H,W], enforce batch=1 here
        if self.enforce_batch1 and batch.shape[0] != 1:
            batch = batch[:1]

        self.batches_yielded += 1
        return {self.input_name: batch}

    def rewind(self):
        self.iterator = iter(self.dataloader)
        self.batches_yielded = 0

reader = CalibrationReader(calib_loader, input_name=input_name, max_batches=num_calib_batches, enforce_batch1=True)

model_fp32 = f"{model_name}.onnx"
model_int8 = f"{model_name}_quant.onnx"

quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=reader,
    calibrate_method=CalibrationMethod.MinMax,
    per_channel=False,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=[],
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    reduce_range=False
)
