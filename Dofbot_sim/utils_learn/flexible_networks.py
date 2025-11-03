import torch
import torch.nn as nn
import torch.nn.functional as F

# 残差块
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, activation='ReLU'):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = getattr(nn, activation)()

    def forward(self, x):
        return self.act(x + self.fc(x))

# Highway 块
class HighwayBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, activation='ReLU'):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(dim, dim), # 全连接层
            nn.BatchNorm1d(dim), # 批归一化
            getattr(nn, activation)(), # 激活函数
            nn.Dropout(dropout), # Dropout
            nn.Linear(dim, dim), # 全连接层
            nn.BatchNorm1d(dim), # 批归一化
        )
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid()) # 门控层
        self.act = getattr(nn, activation)() # 激活函数

    def forward(self, x):
        T = self.act(self.transform(x))
        G = self.gate(x)
        return G * T + (1 - G) * x

# MLP
class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, output_dim,
                 hidden_layers=[256, 256, 256],
                 dropout=0.0, use_bn=True, activation='ReLU',
                 block_type=None,      # 新增：'res' | 'highway' | None
                 num_blocks=1):        # 每个隐藏层后插几个块
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim)) # 全连接层
            if use_bn: layers.append(nn.BatchNorm1d(h_dim)) # 批归一化
            layers.append(getattr(nn, activation)()) # 激活函数
            if dropout > 0: layers.append(nn.Dropout(dropout)) # Dropout

            # 插入残差/Highway 块
            for _ in range(num_blocks):
                if block_type == 'res':
                    layers.append(ResBlock(h_dim, dropout, activation)) # 残差块
                elif block_type == 'highway':
                    layers.append(HighwayBlock(h_dim, dropout, activation)) # Highway 块

            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim)) # 线性映射到目标维度
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 混合密度网络
class FlexibleMDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures=5,
                 hidden_layers=[256, 256, 256], dropout=0.1, use_bn=True, activation='ReLU'):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.output_dim   = output_dim
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim)) # 全连接层
            if use_bn: layers.append(nn.BatchNorm1d(h_dim)) # 批归一化
            layers.append(getattr(nn, activation)()) # 激活函数
            if dropout > 0: layers.append(nn.Dropout(dropout)) # Dropout
            in_dim = h_dim
        self.hidden = nn.Sequential(*layers)

        self.pi_layer    = nn.Linear(in_dim, num_mixtures) # 混合系数层
        self.mu_layer    = nn.Linear(in_dim, num_mixtures * output_dim) # 均值层
        self.sigma_layer = nn.Linear(in_dim, num_mixtures * output_dim) # 方差层

    def forward(self, x):
        h = self.hidden(x)
        pi    = F.softmax(self.pi_layer(h), dim=1) # 混合系数归一化
        mu    = self.mu_layer(h).view(-1, self.num_mixtures, self.output_dim) # 均值重塑
        sigma = torch.exp(self.sigma_layer(h).view(-1, self.num_mixtures, self.output_dim)) # 方差指数映射
        return pi, mu, sigma

    def forward_pred(self, x):
        """测试用：加权平均期望输出 [B, D]"""
        pi, mu, sigma = self.forward(x)
        pi = pi.unsqueeze(-1)                      # [B, K, 1]
        return (pi * mu).sum(dim=1)                # [B, D]

# LSTM 网络
class FlexibleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 output_dim=None, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.) # LSTM 层
        self.output_dim = output_dim
        if output_dim:
            self.fc = nn.Linear(hidden_dim, output_dim) # 线性映射到目标维度
        else:
            self.fc = None

    def forward(self, x):
        out, _ = self.lstm(x)              # [B, T, hidden]
        last   = out[:, -1, :]             # [B, hidden]
        return self.fc(last) if self.fc else last