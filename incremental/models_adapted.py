import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class OpenEmbed(nn.Module):
    def __init__(self, output, vocab_size=256, embedding_dim=128):
        super(OpenEmbed, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 2), stride=2),
            nn.Dropout(0.1)
        )

        # 全连接层
        self.embed = nn.Sequential(
            nn.Linear(2816, 256),
        )
        self.fc = nn.Linear(256, output)  # 分类头
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 输入形状处理: 确保是 (batch_size, 784)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.get_mask(x)
        x = x.unsqueeze(1)  # 增加通道维度
        x = self.conv(x.float())
        x = x.view(x.size(0), -1)
        embed = self.embed(self.dropout(x))
        out = self.fc(embed)
        return embed, out

    def get_mask(self, x):
        # 输入 x: (batch_size, seq_len=784)
        mask = torch.where(x == 255, 0, 1)
        embed_data = self.embedding(x.long())

        # 计算每个序列的真实长度
        seq_lengths = mask.sum(dim=1)

        # 构造一个索引矩阵，用于选择需要置零的位置
        idx = torch.arange(embed_data.size(1)).unsqueeze(0).to(embed_data.device)

        # 将超过真实长度的位置置零
        embed_data[idx >= seq_lengths.unsqueeze(1)] = 0
        return embed_data