import torch.nn as nn
import torch.nn.functional as F

class Pairwise(nn.Module):
  """Pairwise CNN to get features from image"""
  def __init__(self):
    """Summary
      A total of 3 layers with 64 kernels
      Tanh and absolute activation function
      Output has weights for horizontal and vertical orientation
    """
    super(Pairwise, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 2, 3, padding=1)

  def forward(self, x):
    x = F.tanh(self.conv1(x))
    x = F.tanh(self.conv2(x))
    x = self.conv3(x).abs_()
    return x