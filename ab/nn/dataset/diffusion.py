import torch
import torch.nn as nn
import torch.optim as optim
def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class UNet2DModel(nn.Module):
    """
    Simplified UNet2DModel implementation for image classification.
    """

    def __init__(
        self,
        sample_size=32,
        in_channels=3,
        out_channels=128,  # Reduced final output channels
        layers_per_block=2,
        block_out_channels=(32, 64, 128),  # Reduced block output channels
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ):
        super(UNet2DModel, self).__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block

        # Define down blocks
        self.down_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch, block_type in zip(block_out_channels, down_block_types):
            self.down_blocks.append(self._make_block(in_ch, out_ch, block_type))
            in_ch = out_ch

        # Define up blocks
        self.up_blocks = nn.ModuleList()
        for out_ch, block_type in zip(block_out_channels[::-1], up_block_types):
            self.up_blocks.append(self._make_block(in_ch, out_ch, block_type))
            in_ch = out_ch

        # Final conv layer to output channels
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def _make_block(self, in_channels, out_channels, block_type):
        """
        Creates a single block (down or up) with optional attention.
        """
        if block_type.startswith("Down"):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        elif block_type.startswith("Up"):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2
                ),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:  # Default basic block
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

    def forward(self, x, timesteps):
        """
        Forward pass through UNet2DModel.
        """
        # Down sampling
        down_features = []
        for block in self.down_blocks:
            x = block(x)
            down_features.append(x)

        # Up sampling
        for block in self.up_blocks:
            x = block(x)

        # Final output
        x = self.final_conv(x)
        return nn.Identity()(x)


 

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prms):
        super(Net, self).__init__()

        # Extracting information from in_shape and out_shape
        batch = in_shape[0]
        channel_number = in_shape[1]
        image_size = in_shape[2]  # Assuming square images
        class_number = out_shape[0]

        # Initialize the UNet model
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=channel_number,
            out_channels=128,  # Reduced final output channels
            layers_per_block=2,
            block_out_channels=(32, 64, 128),  # Reduced block output channels
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

        # Classifier layer after UNet
        self.classifier = nn.Sequential(
            nn.Dropout(prms.get('dropout', 0.5)),  # Dropout before classifier
            nn.Linear(128, class_number)  # Adjusted to match reduced output channels
        )

        # Set learning rate from prms
        self.learning_rate = prms.get('lr', 0.001)  # Default to 0.001 if not provided

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]

        timesteps = torch.full(
            (batch_size,), 50, dtype=torch.long, device=x.device
        )

        # Pass through UNet model
        unet_output = self.unet(x, timesteps)

        # Convert the output of UNet to float32 before passing it to the classifier
        unet_output = unet_output.to(torch.float32)

        # Apply global average pooling
        pooled_features = unet_output.mean(dim=(2, 3))

        logits = self.classifier(pooled_features)

        return logits

    def train_setup(self, device, prms):
        """ Initialize loss function and optimizer. """
        self.device = device
        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        
        # Access the learning rate from prms or the attribute
        lr = prms.get('lr', self.learning_rate)
        momentum = prms.get('momentum', 0.9)
        
        # Initialize the optimizer with the learning rate
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=lr,
            momentum=momentum
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


