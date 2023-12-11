import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) 

    def forward(self, X):
        return self.embedding(X) * math.sqrt(self.d_model) 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        PE = torch.zeros(seq_len, d_model) 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        PE = PE.unsqueeze(0)
        self.register_buffer("PE", PE)

    def forward(self, X):
        X = X + (self.PE[:, :X.shape[1], :]).requires_grad_(False)
        return self.dropout(X)
    
class ResidualConnection(nn.Module):
    def __init__(self, num_features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(num_features)

    def forward(self, X, sublayer):
        return X + self.dropout(sublayer(self.norm(X))) 

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
    
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def attention(Q, K, V, mask, dropout):
        d_k = Q.shape[-1]
        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k) 

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) 
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ V)

    def forward(self, Q, K, V, mask):

        query = self.w_q(Q)
        key = self.w_k(K)
        value = self.w_v(V)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        X = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) 
        X = X.transpose(1, 2).contiguous().view(X.shape[0], -1, self.h * self.d_k)
        
        return self.w_o(X) 

class LayerNormalization(nn.Module): 
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, X):

        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)

        return self.alpha * (X - mean) / (std + self.eps) + self.bias 
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(X))))
    
class EncoderBlock(nn.Module):
    def __init__(self, num_features, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(num_features, dropout) for _ in range(2)])

    def forward(self, X, src_mask):

        X = self.residual_connections[0](X, lambda X: self.self_attention_block(X, X, X, src_mask)) 
        X = self.residual_connections[1](X, self.feed_forward_block) 

        return X
    
class Encoder(nn.Module):
    def __init__(self, num_features, layers):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(num_features)

    def forward(self, X, mask):

        for layer in self.layers:
            X = layer(X, mask)

        return self.norm(X)
    
class DecoderBlock(nn.Module):
    def __init__(self, num_features, self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(num_features, dropout) for _ in range(3)])

    def forward(self, X, encoder_output, src_mask, trgt_mask):

        X = self.residual_connections[0](X, lambda X: self.self_attention_block(X, X, X, trgt_mask)) 
        X = self.residual_connections[1](X, lambda X: self.cross_attention_block(X, encoder_output, encoder_output, src_mask)) 
        X = self.residual_connections[2](X, self.feed_forward_block)
        
        return X
    
class Decoder(nn.Module):
    def __init__(self, num_features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(num_features)

    def forward(self, X, encoder_output, src_mask, trgt_mask):

        for layer in self.layers:
            X = layer(X, encoder_output, src_mask, trgt_mask)

        return self.norm(X)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        return self.proj(X)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, trgt_embed: InputEmbeddings, src_pos: PositionalEncoding, trgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trgt_embed = trgt_embed
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, trgt, trgt_mask):
        # (batch, seq_len, d_model)
        trgt = self.trgt_embed(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_output, src_mask, trgt_mask)
    
    def project(self, X):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(X)
    
def build_transformer(src_vocab_size, trgt_vocab_size, src_seq_len, trgt_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trgt_embed = InputEmbeddings(d_model, trgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trgt_pos = PositionalEncoding(d_model, trgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, trgt_vocab_size)
 
    transformer = Transformer(encoder, decoder, src_embed, trgt_embed, src_pos, trgt_pos, projection_layer)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
