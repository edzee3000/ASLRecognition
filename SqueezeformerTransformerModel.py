from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *



class SqueezeformerTransformerModel(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model = d_model, num_heads=8,
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







