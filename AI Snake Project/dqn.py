import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.input = nn.Linear(input_dim, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.output(x)
        return x
    
class DQN_CNN(nn.Module):
  def __init__(self, input_channels, extra_dim, output_dim):
      super(DQN_CNN, self).__init__()

      self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=4, padding=1)
      self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
      self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

      self.global_pool = nn.AdaptiveAvgPool2d((1,1))

      self.fc1 = nn.Linear(256 + extra_dim, 256) # 64 128
      self.fc2 = nn.Linear(256, output_dim) # 128

  def forward(self, image, extra):
    x = F.relu(self.conv1(image))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = self.global_pool(x)          # shape becomes (batch_size, 64, 1, 1)
    x = x.view(x.size(0), -1)        # shape: (batch_size, 64)
    x = torch.cat((x, extra), dim=1) # concatenate along the feature dimension
    x = F.relu(self.fc1(x))
    q_values = self.fc2(x)
    return q_values

