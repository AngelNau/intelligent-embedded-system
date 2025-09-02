import torch
import torch.nn as nn
import torch.optim as optim
import train as t
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import utils
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1   = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0 

        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

data_dir = "<>"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.PILToTensor()
    # transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(data_dir, transform=transform)
test_dataset = datasets.ImageFolder(data_dir, transform=transform)
val_dataset = datasets.ImageFolder(data_dir, transform=transform)
train, test, val = utils.getSubsets(train_dataset, test_dataset, val_dataset)
train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test, batch_size=32, shuffle=False, num_workers=2)
val_loader = DataLoader(dataset=val, batch_size=32, shuffle=False, num_workers=2)

model = TinyCNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6
)

history, best_state, (test_loss, test_acc) = t.train_validate_test(model, train_loader, val_loader, criterion, optimizer, device, test_loader=test_loader, num_epochs=1, scheduler=scheduler, early_stop_patience=5, min_delta=1e-4)
model.load_state_dict(best_state)
model = nn.Sequential(model, nn.Softmax(dim=1))
torch.save(model.state_dict(), "tinyCnn_10.pth")
# input_tensor = torch.randn(1, 3, 128, 128)
input_tensor = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8)
torch.onnx.export(
    model,
    (input_tensor,),
    "tinyCnn_10.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=False,
    export_params=True,
    do_constant_folding=True,
    external_data=False,
    opset_version=11
)
