import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


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
        x = x + self.pe[:x.size(0), :]
        return x


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super(ConvModule, self).__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = DepthwiseSeparableConv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.layer_norm(x)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.dropout(x)
        return x


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x


class SqueezeformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size=31, dim_feedforward=2048, dropout=0.1):
        super(SqueezeformerBlock, self).__init__()
        self.mhsa = MultiHeadedSelfAttentionModule(d_model, num_heads, dropout)
        self.feed_forward1 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.conv_module = ConvModule(d_model, kernel_size, dropout)
        self.feed_forward2 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mhsa(x)
        x = x + self.feed_forward1(x)
        x = x + self.conv_module(x)
        x = x + self.feed_forward2(x)
        x = self.layer_norm(x)
        return x


class SqueezeformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=512, num_blocks=12, num_heads=8, kernel_size=31, dim_feedforward=2048,
                 dropout=0.1):
        super(SqueezeformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            SqueezeformerBlock(d_model, num_heads, kernel_size, dim_feedforward, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, num_layers=6, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output


class BERTEmbedding(nn.Module):
    def __init__(self, freeze_bert=True):
        super(BERTEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        return outputs.last_hidden_state

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)


class SqueezeformerTransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, d_model=512, num_encoder_blocks=12, num_decoder_layers=6, num_heads=8, kernel_size=31,
                 dim_feedforward=2048, dropout=0.1, freeze_bert=True):
        super(SqueezeformerTransformerDecoderModel, self).__init__()
        self.encoder = SqueezeformerEncoder(input_dim, d_model, num_encoder_blocks, num_heads, kernel_size,
                                            dim_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, num_decoder_layers, num_heads, dim_feedforward, dropout)
        self.bert_embedding = BERTEmbedding(freeze_bert)

        # 投影层，将模型输出映射到BERT嵌入空间
        self.output_projection = nn.Linear(d_model, 768)  # BERT的隐藏维度是768

        # 特殊token的ID
        self.pad_token_id = 0
        self.bos_token_id = 101  # [CLS]
        self.eos_token_id = 102  # [SEP]

    def forward(self, src, tgt_input_ids):
        # 编码输入视频帧
        encoder_output = self.encoder(src)
        encoder_output = encoder_output.permute(1, 0, 2)  # [T, B, D]

        # 获取BERT嵌入作为目标表示
        tgt_embeddings = self.bert_embedding(tgt_input_ids)
        tgt_embeddings = tgt_embeddings.permute(1, 0, 2)  # [T, B, D]

        # 创建自回归掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embeddings.size(0)).to(src.device)

        # 解码
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=tgt_mask)

        # 投影到BERT嵌入空间
        decoder_output = self.output_projection(decoder_output)

        return decoder_output.permute(1, 0, 2)  # [B, T, D]

    def generate(self, src, max_length=50):
        batch_size = src.size(0)
        encoder_output = self.encoder(src)
        encoder_output = encoder_output.permute(1, 0, 2)  # [T, B, D]

        # 初始化输出序列，以[CLS]开始
        output = torch.ones(batch_size, 1).fill_(self.bos_token_id).long().to(src.device)

        for i in range(max_length - 1):
            # 获取当前输出的BERT嵌入
            tgt_embeddings = self.bert_embedding(output)
            tgt_embeddings = tgt_embeddings.permute(1, 0, 2)  # [T, B, D]

            # 创建自回归掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embeddings.size(0)).to(src.device)

            # 解码
            decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=tgt_mask)

            # 投影到BERT嵌入空间
            decoder_output = self.output_projection(decoder_output)  # [T, B, 768]

            # 计算与BERT词汇表中每个词的相似度
            bert_embeddings = self.bert_embedding.bert.embeddings.word_embeddings.weight  # [vocab_size, 768]
            similarities = torch.matmul(decoder_output[-1], bert_embeddings.transpose(0, 1))  # [B, vocab_size]

            # 获取最相似的词
            next_token = torch.argmax(similarities, dim=-1).unsqueeze(1)

            # 添加到输出序列
            output = torch.cat([output, next_token], dim=1)

            # 如果所有序列都遇到了[SEP]，则停止生成
            if (next_token == self.eos_token_id).all():
                break

        return output


def create_tgt_mask(tgt):
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
    return tgt_mask


def prepare_data(df):
    frames = []
    for frame in range(df['frame'].max() + 1):
        frame_data = df[df['frame'] == frame]
        # 提取所有特征列（除了frame列）
        features = frame_data.drop('frame', axis=1).values.flatten()
        frames.append(features)

    # 转换为PyTorch张量
    return torch.FloatTensor(np.array(frames))


class VideoPhraseDataset(Dataset):
    def __init__(self, dataframes, phrases, bert_tokenizer):
        self.dataframes = dataframes
        self.phrases = phrases
        self.tokenizer = bert_tokenizer

    def __len__(self):
        return len(self.dataframes)

    def __getitem__(self, idx):
        df = self.dataframes[idx]
        phrase = self.phrases[idx]

        # 准备视频特征
        video_features = prepare_data(df)

        # 准备文本标记
        encoded = self.tokenizer(phrase, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            'video_features': video_features,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def collate_fn(batch):
    video_features = [item['video_features'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    # 对视频特征进行填充，使它们具有相同的帧数
    video_features = pad_sequence(video_features, batch_first=True)

    # 将其他张量转换为批次形式
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)

    return {
        'video_features': video_features,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            video_features = batch['video_features'].to(device)
            input_ids = batch['input_ids'].to(device)

            # 移除最后一个token（通常是[SEP]），因为我们不需要预测它
            tgt_input_ids = input_ids[:, :-1]

            # 目标是下一个token的BERT嵌入
            target_embeddings = model.bert_embedding(input_ids)[:, 1:, :]  # 从第二个token开始

            optimizer.zero_grad()

            # 前向传播
            output = model(video_features, tgt_input_ids)

            # 计算损失
            loss = criterion(output, target_embeddings)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 示例输入：103帧，每帧1630个关键点特征
    batch_size = 4
    input_dim = 1630
    seq_length = 103

    # 创建随机输入数据模拟关键点检测结果
    src = torch.randn(batch_size, seq_length, input_dim).to(device)

    # 初始化模型
    model = SqueezeformerTransformerDecoderModel(
        input_dim=input_dim,
        d_model=512,
        num_encoder_blocks=6,
        num_decoder_layers=6,
        num_heads=8,
        kernel_size=31,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)

    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 示例目标短语
    phrases = [
        "a person is walking",
        "someone is sitting on a chair",
        "a man is running",
        "a woman is dancing"
    ]

    # 准备训练数据
    dfs = []
    for _ in range(batch_size):
        # 创建示例DataFrame
        columns = ['frame']
        for prefix in ['x_face', 'x_left_hand', 'x_pose', 'x_right_hand', 'y_face', 'y_left_hand', 'y_pose',
                       'y_right_hand', 'z_face', 'z_left_hand', 'z_pose', 'z_right_hand']:
            columns.extend([f"{prefix}_{i}" for i in range(100)])  # 假设每个部分有100个点

        data = np.random.rand(103, 1 + 9 * 100)  # 103帧，每帧1+900个特征
        df = pd.DataFrame(data, columns=columns)
        df['frame'] = df['frame'].astype(int) * 103
        dfs.append(df)

    # 创建数据集和数据加载器
    dataset = VideoPhraseDataset(dfs, phrases, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    train(model, dataloader, optimizer, criterion, device, epochs=5)

    # 生成文本示例
    model.eval()
    with torch.no_grad():
        generated = model.generate(src)

    # 解码生成的ID为文本
    for i in range(batch_size):
        tokens = tokenizer.convert_ids_to_tokens(generated[i].tolist())
        print(f"Generated: {tokenizer.convert_tokens_to_string(tokens)}")


if __name__ == "__main__":
    main()