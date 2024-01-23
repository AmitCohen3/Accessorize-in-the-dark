import torch.nn as nn
import torch.nn.functional as F

class LowRankEmbedding(nn.Module):
    def __init__(self):
        super(LowRankEmbedding, self).__init__()
        self.emb = nn.Linear(4096,4096)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))