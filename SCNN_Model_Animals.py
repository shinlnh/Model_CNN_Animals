import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_class = 10):

        super().__init__()

        self.block_conv1 = self.make_block_conv(in_channels=3,out_channels=8)
        self.block_conv2 = self.make_block_conv(in_channels=8,out_channels=16)
        self.block_conv3 = self.make_block_conv(in_channels=16,out_channels=32)
        self.block_conv4 = self.make_block_conv(in_channels=32,out_channels=64)
        self.block_conv5 = self.make_block_conv(in_channels=64,out_channels=128)
        self.flatten = nn.Flatten()
        self.fc1 = self.make_fc_layer(in_features=6272,out_features=256)
        self.fc2 = self.make_fc_layer(in_features=256,out_features=512)
        self.fc3 = self.make_fc_layer(in_features=512,out_features=1024)
        self.fc4 = self.make_fc_layer(in_features=1024,out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=num_class)



    def make_block_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    def make_fc_layer(self, in_features,out_features):
        return nn.Sequential(

            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.LeakyReLU()
        )


    def forward(self, x):
        x = self.block_conv1(x)
        x = self.block_conv2(x)
        x = self.block_conv3(x)
        x = self.block_conv4(x)
        x = self.block_conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        return x

if __name__ == '__main__':
    model = SimpleCNN()
    input_data = torch.rand(8,3,224,224)
    result = model(input_data)
    print(result)
    print(result.shape)
