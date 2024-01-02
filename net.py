import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# 1.full connected network
class Full_Net(nn.Module):
    def __init__(self):
        """
        Initialize the neural network model.
        """
        super(Full_Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 2.convolutional neural network for number recognition
class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()

        self.conv_1=nn.Sequential(
            # 1. input 1*28*28 output 16*28*28
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # 2. batch normalization
            nn.BatchNorm2d(16),
            # 3. activation function
            nn.ReLU(),
            # 4. max pooling and output 16*14*14 
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.conv_2=nn.Sequential(
            # 卷积 输入：bs*16*14*14  输出：bs*32*14*14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # 归一化
            nn.BatchNorm2d(32),
            # 激活函数
            nn.ReLU(),
            # 最大池化：输入:bs*32*14*14  输出：bs*32*7*7
            nn.MaxPool2d(2)
        )
        # 第三层卷积，输入：bs*32*7*7 输出：bs*64*3*3
        self.conv_3 = nn.Sequential(
            # 卷积 输入：bs*32*7*7  输出：bs*64*3*3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 归一化
            nn.BatchNorm2d(64),
            # 激活函数
            nn.ReLU(),
            # 最大池化：输入：bs*64*7*7 输出：bs*64*3*3
            nn.MaxPool2d(2)
        )
        # 自适应池化，将bs*64*3*3映射为bs*64*1*1
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
         # 全连接层
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # 1. conv layer
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        # 2. adaptive pooling
        x = self.advpool(x)
        # 3. flatten
        x = x.view(x.size(0), -1)
        # 4. fc layer
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # 1. create the model
    model = Conv_Net()
    # 2. create the writer for tensorboardX
    writer = SummaryWriter("./pytorch_tb")
    # 3. write the graph to tensorboardX
    writer.add_graph(model, torch.rand(1, 1, 28, 28))
    writer.close()
