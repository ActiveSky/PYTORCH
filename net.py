
import torch.nn as nn
import torch.nn.functional as F



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

import torch.nn.functional as F
   
# 2.convolutional neural network
class Conv_Net(nn.Module):
   def __init__(self):
       """
       Initialize the neural network model.
       """
       super(Conv_Net, self).__init__()
       self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
       self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
       self.conv2_drop = nn.Dropout2d()
       self.fc1 = nn.Linear(320, 50)
       self.fc2 = nn.Linear(50, 10)

   def forward(self, x):
       """
       Perform forward pass through the neural network.

       Args:
           x (torch.Tensor): Input tensor.

       Returns:
           torch.Tensor: Output tensor.
       """
       x = F.relu(F.max_pool2d(self.conv1(x), 2))
       x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
       x = x.view(-1, 320)
       x = F.relu(self.fc1(x))
       x = F.dropout(x, training=self.training)
       x = self.fc2(x)
       return x
