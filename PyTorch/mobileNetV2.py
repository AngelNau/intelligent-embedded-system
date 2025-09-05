import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import utils
import weights as w
import train as t


class WrappedModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone   # store the original model

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0 

        out = self.backbone(x)
        return out


data_dir = "../flower_photos/"
model_name = "mobileNetV2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.PILToTensor(),
    # transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(data_dir, transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(data_dir, transform=data_transforms)
val_dataset = torchvision.datasets.ImageFolder(data_dir, transform=data_transforms)
train, test, val = utils.getSubsets(train_dataset, test_dataset, val_dataset)
train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test, batch_size=32, shuffle=False, num_workers=2)
val_loader = DataLoader(dataset=val, batch_size=32, shuffle=False, num_workers=2)

pre_trained_model = torchvision.models.mobilenet_v2(weights="DEFAULT").state_dict()
model = torchvision.models.mobilenet_v2(width_mult=0.35)
w.transfer_and_slice(pre_trained_model, model)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=model.last_channel, out_features=5, bias=True)
)
model = WrappedModel(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)

history, best_state, (test_loss, test_acc) = t.train_validate_test(model, train_loader, val_loader, criterion, optimizer, device, test_loader=test_loader, num_epochs=30, scheduler=scheduler)
model.load_state_dict(best_state)
model = nn.Sequential(model, nn.Softmax(dim=1))
model.cpu()
model.eval()
# input_tensor = torch.randn(1, 3, 128, 128)
input_tensor = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8)
torch.onnx.export(
    model,
    (input_tensor,),
    f"{model_name}.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=False,
    export_params=True,
    do_constant_folding=True,
    external_data=False,
    opset_version=11
)
