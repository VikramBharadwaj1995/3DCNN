from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
import torch.nn as nn
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(in_features = 4*8*8*256, out_features = 256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features = 256, out_features = 2)
      
    def forward(self, s):
      s = self.bn1(self.conv1(s))        # batch_size x 32 x 64 x 64
      #print(s.shape)
      s = F.relu(F.max_pool2d(s, 2))     # batch_size x 32 x 32 x 32
      #print(s.shape)
      s = self.bn2(self.conv2(s))        # batch_size x 64 x 32 x 32
      #print(s.shape)
      s = F.relu(F.max_pool2d(s, 2))     # batch_size x 64 x 16 x 16
      #print(s.shape)
      s = self.bn3(self.conv3(s))        # batch_size x 128 x 16 x 16
      #print(s.shape)
      s = F.relu(F.max_pool2d(s, 2))     # batch_size x 128 x 8 x 8
      #print(s.shape)
      s = self.bn4(self.conv4(s))        # batch_size x 128 x 16 x 16
      #print(s.shape)
      s = F.relu(F.max_pool2d(s, 2))     # batch_size x 128 x 8 x 8
      

      #flatten the output for each image
      s = s.view(s.size(0), -1)  # batch_size x 8*8*128
      #print(s.shape)

      #apply 2 fully connected layers with dropout
      s = F.relu(self.fcbn1(self.fc1(s)))
      s = self.fc2(s)

      return F.log_softmax(s, dim=1)