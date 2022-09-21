from torch import nn

class Net(nn.Module):
    
    def __init__(self, n_classes=10):
        super().__init__()  

        # Note: Tensors (arrays) here are of shape [B, C, M, N], where:
        # B is the number of samples in a batch
        # C is the number of "color" or feature channels
        # M is the first spatial dimension
        # N is the second spatial dimension


        # Input is [B, 3, 64, 64]
        self.convolutions = nn.Sequential( 
            nn.Conv2d(3, 32, 3, padding=1), # [B, 32, 64, 64]
            nn.BatchNorm2d(32), # [B, 32, 64, 64]
            nn.ReLU(inplace=True), # [B, 32, 64, 64]
            nn.MaxPool2d(2, 2), # [B, 32, 32, 32]
            #==============================================================
            nn.Conv2d(32, 64, 3, padding=1), # [B, 64, 32, 32]
            nn.BatchNorm2d(64), # [B, 64, 32, 32]
            nn.ReLU(inplace=True), # [B, 64, 32, 32]
            nn.MaxPool2d(2, 2), # [B, 64, 16, 16]
        )

        # Input will be reshaped from [B, 64, 16, 16] to [B, 64*16*16] for fully connected layers
        self.fully_connected = nn.Sequential(
            nn.Linear(64*16*16, 256), # [B, 256]
            nn.ReLU(inplace=True), # [B, 256]
            nn.Linear(256, n_classes), # [B, n_classes]
        )

        # Note: the final output must have shape [B, n_classes]

        # We're skipping a softmax activation here since we'll be using a loss function that does it automatically



    def forward(self, img):

        # Apply convolution operations
        x = self.convolutions(img)

        # Reshape
        x = x.view(x.size(0), -1)

        # Apply fully connected operations
        x = self.fully_connected(x)

        return x