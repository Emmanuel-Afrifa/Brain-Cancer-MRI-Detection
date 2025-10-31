from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, BatchNorm1d, Dropout, Linear
from torchinfo import summary
import torch.nn.functional as F
import torch

class BrainScanCNN(torch.nn.Module):
    """
    This method inherits from the torch.nn.Module and builds the base model for this project.

    Attributes:
        num_classes (int): 
            Number of neurons in the output layer
        model_name (str):
            Name of the model
            
    Methods:
        forward(self, x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
            
        __str__(self) -> str:
            Defines how the model should be printed.
    """
    def __init__(self, num_classes: int, model_name: str = "Brain Scan CNN") -> None:
        super().__str__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = MaxPool2d(kernel_size=2)
        self.bn1 = BatchNorm2d(16)
        self.bn2 = BatchNorm2d(32)
        self.bn3 = BatchNorm2d(64)
        self.bn4 = BatchNorm2d(128)
        self.bn5 = BatchNorm2d(256)
        self.fc1 = Linear(in_features=256*7*7, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=128)
        self.fc3 = Linear(in_features=128, out_features=self.num_classes)
        self.dropout1 = Dropout()
        self.dropout2 = Dropout()
        self.bn_fc1 = BatchNorm1d(256)
        self.bn_fc2 = BatchNorm1d(128)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = torch.flatten(x)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        
        return x
    
    def __str__(self) -> str:    
        try:
            model_summary = summary(
                self,
                input_size=(32, 3, 224, 224)
            )
            return str(model_summary)
        except Exception as e:
            return f"{self.model_name}(input_shape={[32, 3, 224, 224]}, num_classes={self.num_classes})\n[Summary failed: {e}]"