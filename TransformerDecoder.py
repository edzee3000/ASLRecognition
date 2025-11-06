from utils import *



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=dropout):
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
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=1000, dropout=dropout):
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



