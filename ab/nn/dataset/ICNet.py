import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Function to return the supported hyperparameters
def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

# Initial Block for ICNet
class ICInitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ICInitBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

# Pyramid Spatial Pooling Block
class PSPBlock(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PSPBlock, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False)
            )
            for size in pool_sizes
        ])
        self.bottleneck = nn.Conv2d(in_channels + in_channels // len(pool_sizes) * len(pool_sizes), in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        pooled_features = [F.interpolate(stage(x), size=size, mode='bilinear', align_corners=False) for stage in self.stages]
        return F.relu(self.bottleneck(torch.cat([x] + pooled_features, dim=1)))

# ICNet for classification
class ICNetClassification(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(ICNetClassification, self).__init__()
        self.init_block = ICInitBlock(in_channels, 64)
        self.psp_block = PSPBlock(64, pool_sizes=[1, 2, 3, 6])
        self.head_block = nn.Conv2d(64, 128, kernel_size=1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.init_block(x)
        x = self.psp_block(x)
        x = self.head_block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Wrapper Class Net
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prms):
        super(Net, self).__init__()

        # Extracting information from in_shape and out_shape
        batch = in_shape[0]
        channel_number = in_shape[1]
        image_size = in_shape[2]  # Assuming square images
        class_number = out_shape[0]

        # Initialize the model
        self.model = ICNetClassification(num_classes=class_number, in_channels=channel_number)

        # Extracting hyperparameters from prms with default values
        self.learning_rate = prms.get('lr', 0.001)
        self.momentum = prms.get('momentum', 0.9)
        self.dropout = prms.get('dropout', 0.5)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, device, prms):
        """ Initialize loss function and optimizer. """
        self.device = device
        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, train_data):
        """ Training loop for the model. """
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)  # Gradient clipping
            self.optimizer.step()




