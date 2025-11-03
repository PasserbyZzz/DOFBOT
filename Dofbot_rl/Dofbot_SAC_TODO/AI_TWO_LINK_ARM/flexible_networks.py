import torch
import torch.nn as nn
from torch.distributions import Normal

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[256, 256, 256]):
        """
        input_dim: 输入维度
        output_dim: 输出维度
        hidden_layers: 一个 list, 指定每个隐藏层的神经元数量, 例如 [256, 256, 128, 64]
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FlexibleMDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures=5, hidden_layers=[256, 256, 256]):
        """
        input_dim: 输入维度
        output_dim: 输出维度
        num_mixtures: 高斯混合成分数量
        hidden_layers: 隐藏层神经元列表
        """
        super().__init__()
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.hidden = nn.Sequential(*layers)

        self.pi_layer = nn.Linear(in_dim, num_mixtures)
        self.mu_layer = nn.Linear(in_dim, num_mixtures * output_dim)
        self.sigma_layer = nn.Linear(in_dim, num_mixtures * output_dim)

    def forward(self, x):
        h = self.hidden(x)
        pi = nn.functional.softmax(self.pi_layer(h), dim=1)
        mu = self.mu_layer(h).view(-1, self.num_mixtures, self.output_dim)
        sigma = torch.exp(self.sigma_layer(h).view(-1, self.num_mixtures, self.output_dim))
        return pi, mu, sigma

def mdn_loss_fn(pi, mu, sigma, target):
    target = target.unsqueeze(1).expand_as(mu)
    prob = Normal(loc=mu, scale=sigma)
    log_prob = prob.log_prob(target)
    log_prob = log_prob.sum(dim=2)
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)
    return -torch.mean(log_sum)

class FlexibleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=None):
        """
        input_dim: 输入特征维度
        hidden_dim: 每个 LSTM 层的隐藏单元数量
        num_layers: LSTM 堆叠层数
        output_dim: 如果指定，则在 LSTM 后面接一个全连接层映射到 output_dim
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_dim = output_dim
        if output_dim is not None:
            self.fc = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = None

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        output, (hn, cn) = self.lstm(x)
        # 默认返回最后一个时间步的输出
        last_output = output[:, -1, :]  # [batch, hidden_dim]
        if self.fc:
            return self.fc(last_output)
        else:
            return last_output


class FlexibleRBF(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers=50, sigma=None):
        """
        FlexibleRBF: 径向基函数网络
        input_dim: 输入维度
        output_dim: 输出维度
        num_centers: RBF 神经元数量
        sigma: RBF 高斯核宽度 (float 或 None, None 时自动初始化为常数)
        """
        super().__init__()
        self.num_centers = num_centers
        self.output_dim = output_dim

        # 可训练中心
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))

        # 可训练或固定 sigma
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.zeros(1))  # log sigma 可训练
        else:
            self.log_sigma = torch.log(torch.tensor([sigma]))

        # 输出层
        self.linear = nn.Linear(num_centers, output_dim)

    def rbf_layer(self, x):
        """
        x: [batch, input_dim]
        returns: [batch, num_centers]
        """
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        c = self.centers.unsqueeze(0)  # [1, num_centers, input_dim]
        dist_sq = torch.sum((x - c) ** 2, dim=2)  # [batch, num_centers]
        sigma_sq = torch.exp(self.log_sigma * 2)  # sigma^2
        rbf_activations = torch.exp(-dist_sq / (2 * sigma_sq))
        return rbf_activations

    def forward(self, x):
        rbf_out = self.rbf_layer(x)
        return self.linear(rbf_out)
