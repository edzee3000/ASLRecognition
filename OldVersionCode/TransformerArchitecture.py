import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertModel, BertTokenizer
import pyarrow.parquet as pq


max_video_frame_len = 250
max_text_phrase_len = 30
bert_path = './Tokenizer/bert-base-uncased'# 指定BertTokenizer的本地路径
batch_size = 4



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        # print(f"x的形状为：{x.shape}")
        # print(f"self.pe的形状为：{self.pe.shape}")
        # print(f"x.size(0)的值为：{x.size(0)}")
        # print(f"x.size(1)的值为：{x.size(1)}")
        # print(f"self.pe[:x.size(1), :]的形状为：{self.pe[:x.size(1), :].shape}")
        x = x + self.pe[:x.size(1), :]
        return x


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key_value=None, mask=None):
        # 如果没有提供key_value，默认为自注意力（query=key_value）
        if key_value is None:
            key_value = query
        batch_size = query.size(0)
        # 线性投影并分割多头
        k = self.k_linear(key_value).view(batch_size, -1, self.num_heads, self.d_k)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(key_value).view(batch_size, -1, self.num_heads, self.d_k)
        # Transpose to get dimensions batch_size * num_heads * seq_len * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask = mask.unsqueeze(1).unsqueeze(-1)  # Add head dimension
            # print(f"mask的形状为：{mask.shape}")   # 这里打印一下mask形状确认一下
            # print(f"scores的形状为：{scores.shape}")   # 这里打印scores的形状确认一下  确定跟mask可以广播
            scores = scores.masked_fill(mask == 0, -1e9)
        # 应用softmax和加权求和
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        # Concatenate heads and put through final linear layer 合并多头并通过最终线性层
        concat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class SqueezeformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size, d_ff, dropout=0.1):
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
    def __init__(self, input_dim, d_model, num_heads, num_layers, kernel_size, d_ff, max_len=1000, dropout=0.1):
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


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedSelfAttention(d_model, num_heads)
        self.cross_attn = MultiHeadedSelfAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        residual = tgt
        tgt = self.norm1(tgt)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
        # print(f"tgt_mask的形状为：{tgt_mask.shape}")
        tgt = self.self_attn(tgt, mask=tgt_mask)
        tgt = self.dropout1(tgt)
        tgt = residual + tgt
        # print(f"经过 self_attn 之后tgt的形状为：{tgt.shape}")
        # Cross-attention
        residual = tgt
        tgt = self.norm2(tgt)
        memory_mask = memory_mask.unsqueeze(1).unsqueeze(2) # 这里给memory_mask的形状修改一下
        # print(f"memory_mask的形状为：{memory_mask.shape}")
        tgt = self.cross_attn(query=tgt, key_value=memory, mask=memory_mask)
        tgt = self.dropout2(tgt)
        tgt = residual + tgt
        # print(f"经过 Cross-attention 之后tgt的形状为：{tgt.shape}")
        # Feed-forward
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = self.feed_forward(tgt)
        tgt = self.dropout3(tgt)
        tgt = residual + tgt
        # print(f"经过 Feed-forward 之后tgt的形状为：{tgt.shape}")
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=1000, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)
        # print(f"vocab_size的值为：{vocab_size}")
        # print(f"d_model的值为：{d_model}")

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # print(f"token_embedding之前tgt的形状为：{tgt.shape}")
        tgt = self.token_embedding(tgt)
        # print(f"经过token_embedding之后tgt的形状为：{tgt.shape}")
        tgt = self.pos_encoder(tgt)
        # print(f"经过pos_encoder之后tgt形状为：{tgt.shape}")
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        # print(f"经过TransformerDecoderLayers之后tgt形状为：{tgt.shape}")
        tgt = self.norm(tgt)
        output = self.generator(tgt)
        return output


class SqueezeformerTransformerModel(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 kernel_size=31, d_ff=2048, max_len=1000, dropout=0.1):
        super(SqueezeformerTransformerModel, self).__init__()
        self.encoder = SqueezeformerEncoder(
            input_dim, d_model, num_heads, num_encoder_layers,
            kernel_size, d_ff, max_len, dropout
        )
        self.decoder = TransformerDecoder(
            vocab_size, d_model, num_heads, num_decoder_layers,
            d_ff, max_len, dropout
        )
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        # print(f"memory的结果为：{memory}")
        # print(f"经过encoder之后memory的形状为：{memory.shape}")
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        # print(f"经过decoder之后output的形状为：{output.shape}")
        return output
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    def decode(self, memory, tgt, tgt_mask):
        return self.decoder(tgt, memory, tgt_mask)


class VideoTextDataset(Dataset):
    def __init__(self, video_features_list, text_list, tokenizer, max_video_len=max_video_frame_len, max_text_len=max_text_phrase_len):
        self.video_features_list = video_features_list
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.max_video_len = max_video_len
        self.max_text_len = max_text_len
        # Precompute video lengths and paddings
        self.video_lengths = [min(len(video), max_video_len) for video in video_features_list]
    def __len__(self):
        return len(self.video_features_list)
    def __getitem__(self, idx):
        video = self.video_features_list[idx]
        text = self.text_list[idx]
        # Truncate or pad video to max_video_len
        video_len = min(len(video), self.max_video_len)
        if len(video) < self.max_video_len:
            # Pad with zeros
            padded_video = torch.zeros((self.max_video_len, video.shape[1]))
            padded_video[:len(video)] = torch.tensor(video)
        else:
            padded_video = torch.tensor(video[:self.max_video_len])
        # Create src_mask: 1 for valid frames, 0 for padded frames
        src_mask = torch.zeros(self.max_video_len, dtype=torch.bool)
        src_mask[:video_len] = 1
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Remove batch dimension
        input_ids = encoding['input_ids'].squeeze()
        text_mask = encoding['attention_mask'].squeeze()
        # 返回一个字典  对应的名称:对应的数据
        return {
            'video': padded_video,
            'text_input_ids': input_ids,
            'text_attention_mask': text_mask,
            'src_mask': src_mask,
            'video_length': video_len
        }


def create_mask(tgt):
    # Create a mask to hide padding and future words
    seq_len = tgt.size(1)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len))
    return tgt_mask





def create_list():
    """ """
    # 读取 CSV 文件
    dataset_df = pd.read_csv('./Data/450474571.csv')
    print("5414471.csv的形状为：{}".format(dataset_df.shape))
    print(f"csv中前几行数据为：\n{dataset_df.head()}")
    # 提取 video_features_list 和 text_list
    video_features_list = []
    text_list = []
    for index, row in dataset_df.iterrows():
        file_path = row['path']
        sequence_id = row['sequence_id']
        phrase = row['phrase']
        file_id = row['file_id']
        sequence_df = pq.read_table(f"./Data/{str(file_id)}.parquet",
                                           filters=[[('sequence_id', '=', sequence_id)], ]).to_pandas()
        print(f"这里我们取sequence_id为：{sequence_id}")
        print(f"{str(file_id)}.parquet的形状为：{sequence_df.shape}")
        # print(f"{str(file_id)}.parquet的前几行为：\n{sequence_df.head()}")
        sequence_df = sequence_df.fillna(0)  # 填充NaN值为0
        sequence_numpy = sequence_df.to_numpy()
        tensor_array = torch.tensor(sequence_numpy, dtype=torch.float32)
        # 将tensor张量添加到 video_features_list 中
        video_features_list.append(tensor_array)
        text_list.append(phrase)
        if index == 8:
            break
    # print("video_features_list:", video_features_list)
    # print("text_list:", text_list)
    return video_features_list, text_list


# Example usage
def main():
    video_features_list, text_list = create_list()
    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(f"{bert_path}")
    print(f"词汇表tokenizer.vocab_size的大小为：{tokenizer.vocab_size}")
    # Initialize dataset and dataloader
    dataset = VideoTextDataset(video_features_list, text_list, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # Initialize model
    model = SqueezeformerTransformerModel(
        input_dim=1630,
        vocab_size=tokenizer.vocab_size,
        d_model=512
    )
    # assert 0
    # Training loop
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            video = batch['video']
            # print(f"video的形状为：{video.shape}")
            text_input_ids = batch['text_input_ids']
            # print(f"text_input_ids的形状为：{text_input_ids.shape}")
            src_mask = batch['src_mask']
            # print(f"src_mask的形状为：{src_mask.shape}")
            # Create target tensor (shifted by one position)
            tgt_input = text_input_ids[:, :-1]
            # print(f"tgt_input的值为：{tgt_input}")
            # print(f"tgt_input的形状为：{tgt_input.shape}")
            tgt_output = text_input_ids[:, 1:]
            # print(f"tgt_output的形状为：{tgt_output.shape}")
            # Create masks
            tgt_mask = create_mask(tgt_input)
            # print(f"tgt_mask的形状为：{tgt_mask.shape}")
            # Forward pass with src_mask and memory_mask
            output = model(video, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
            # print(f"经过model前向传播之后output的形状为：{output.shape}")
            # Reshape output and target for loss calculation
            output_dim = output.size(-1)
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            # print(f"output的结果为：{output}")
            # print(f"tgt_output结果为：{tgt_output}")
            # assert 0
            # print(f"output的形状为：{output.shape}")
            # print(f"tgt_output的形状为：{tgt_output.shape}")
            # Calculate loss
            loss = criterion(output, tgt_output)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader)}')


if __name__ == "__main__":
    main()