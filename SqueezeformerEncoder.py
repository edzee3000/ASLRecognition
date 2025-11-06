from utils import *







class SqueezeformerBlock(nn.Module):
    """
        SqueezeformerEncoder的Block  为了复用
    """
    def __init__(self, d_model, num_heads, kernel_size, d_ff, dropout=dropout):
        super(SqueezeformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadedSelfAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        # 卷积分支
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = DepthwiseSeparableConv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        # 前馈网络
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        # 挤压激励(SE)模块   挤压：通过全局平均池化提取全局上下文信息。 激励：通过两层全连接网络学习通道间依赖关系，生成自适应权重。  应用：增强重要特征，抑制不重要特征，提升模型表现力。
        self.se_module = nn.Sequential(
            nn.Linear(d_model, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, d_model),
            nn.Sigmoid()    # 这里竟然使用到了Sigmoid函数？？？？
        )

    def forward(self, x, mask=None):
        # Self-attention part
        residual = x
        x = self.norm1(x)
        # print(f"norm1之后的x形状为：{x.shape}")
        # print(f"mask的形状为：{mask.shape}")
        mask = mask.unsqueeze(1).unsqueeze(-1)
        x = self.attention(x, mask=mask)
        x = self.dropout1(x)
        x = residual + x
        # print(f"经过Self-attention之后x的形状为：{x.shape}")

        # Convolution part
        residual = x  # x的形状为[batch, seq_len, embed_dim]
        x = self.norm2(x)
        x = x.transpose(1, 2)  # [B, L, D] -> [B, D, L]
        x = self.conv(x)
        # print(f"经过 Convolution 之后没有旋转前的x形状为：{x.shape}")
        x = x.transpose(1, 2)  # [B, D, L] -> [B, L, D]
        # print(f"经过 Convolution 之后x的形状为：{x.shape}")

        # Squeeze-and-Excitation 挤压激励(SE)模块
        se = torch.mean(x, dim=1)
        se = self.se_module(se).unsqueeze(1)
        x = x * se
        x = self.dropout2(x)
        x = residual + x
        # print(f"经过 Squeeze-and-Excitation 挤压激励之后x的形状为：{x.shape}")

        # Feed-forward part
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = residual + x
        # print(f"经过 Feed-forward 前馈部分之后x的形状为：{x.shape}")
        return x




class SqueezeformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, kernel_size, d_ff, max_len=1000, dropout=dropout):
        super(SqueezeformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)   #input_dim为1630
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            SqueezeformerBlock(d_model, num_heads, kernel_size, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # print(f"x的形状为：{x.shape}")
        # print(f"未经处理的x结果为:{x}")
        x = self.input_projection(x)
        # print(f"经过input_projection之后的x的形状为：{x.shape}")
        # print(f"input_projection的结果为：{x}")
        # assert 0
        x = self.pos_encoder(x)
        # print(f"经过position_encoder位置编码之后x的形状为：{x.shape}")
        for layer in self.layers:
            x = layer(x, mask)
        # print(f"经过多层layers之后x的形状为：{x.shape}")
        x = self.norm(x)
        # print(f"经过LayerNorm层归一化之后x的形状为：{x.shape}")
        return x
