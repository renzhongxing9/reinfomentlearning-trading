import torch.nn as nn

# Neural Network
class Net(nn.Module):
    def __init__(self, n_actions, n_states):
        self.n_states = n_states
        self.n_actions = n_actions
        super(Net, self).__init__()
        self.fc1 = nn.Linear(self.n_states, 128)
         # 初始化权重, 用二值分布来随机生成参数的值, 下同
        self.fc1.weight.data.normal_(0, 0.1)  
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, self.n_actions)
        self.out.weight.data.normal_(0, 0.1)
        self.softmax_out = nn.Softmax(dim=1)
    # 前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.out(x)
        actions_value = self.softmax_out(x)
        return actions_value