import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

class MinimalCNN(nn.Module):
    """Custom CNN for ArUco marker classification.
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    Architecture:
    - 3 conv blocks (conv->bn->relu->maxpool)
    - Global average pooling
    - Fully connected layer with dropout
    """
    def __init__(self, num_classes=100):
        super(MinimalCNN, self).__init__()
        # Initial conv block: RGB input -> 32 features
        # (RGB so that it matches with the other models and the input doesn't have to be changed twice)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block: 32 -> 64 features
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third conv block: 64 -> 128 features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply conv blocks with maxpool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def get_model(model_name, num_classes=100):
    """Returns classification model based on specified architecture.
    https://pytorch.org/vision/main/models.html
    
    Models:
    - MinimalCNN: Custom CNN
    - AlexNet-clean: AlexNet without pretrained weights
    - AlexNet: Pretrained with frozen features
    - ResNet18: Pretrained with frozen features
    - GoogLeNet: Pretrained with frozen features
    """
    if model_name == "MinimalCNN":
        return MinimalCNN(num_classes=num_classes)
    elif model_name == "AlexNet-clean":
        model = models.alexnet()
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_name == "AlexNet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        #model.features.requires_grad_ = True  # Freeze feature extraction
        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == "GoogLeNet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_detection_model(model_name, num_classes=2):
    """Returns detection model for ArUco marker detection.
    https://pytorch.org/vision/0.20/_modules/torchvision/models/detection/retinanet.html
    https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/faster_rcnn.html
    https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html
    
    Models:
    - RetinaNet-ResNet50: ~40M parameters.
    - FasterRCNN-ResNet50: ~38M parameters.
    - MobileNetV3-Large-FPN: ~20M parameters.
    """
    if model_name == "RetinaNet-ResNet50":
        model = retinanet_resnet50_fpn(weights="DEFAULT")
        
        # Modify classification head for binary detection
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes
        )
        return model

    elif model_name == "FasterRCNN-ResNet50":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        
        # Replace RoI head classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    elif model_name == "MobileNetV3-Large-FPN":
        model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        
        # Modify for binary detection
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")
