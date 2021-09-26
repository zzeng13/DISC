"""
Implement the BI-DIRECTIONAL ATTENTION FLOW model.

Paper reference: https://arxiv.org/pdf/1611.01603.pdf
Code reference: https://github.com/galsang/BiDAF-pytorch

    NOTE:
        As in the paper:
            N = batch_size
            T = max_len_context
            J = max_len_query

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.model.lstm import LSTM
# from src.model.bilstm import EncoderBiLSTM, DecoderLSTMWithAttention


# Contextual RNN (For Contextual Embedding)
# ------------------------------------------------------------
class ContextualEmbeddingLayer(nn.Module):
    """
    In: ((N, seq_len, emb_dim), max_len_context)
    Out: (N, seq_len, lstm_hidden_dim)
    """
    def __init__(self, config):
        super(ContextualEmbeddingLayer, self).__init__()

        self.context_LSTM = nn.LSTM(input_size=config.EMBEDDING_DIM,
                                 hidden_size=config.LSTM_HIDDEN_DIM,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=config.LSTM_DROP_RATE)
        self.dropout = nn.Dropout(config.LSTM_DROP_RATE)

    def forward(self, x):
        x = self.dropout(x)
        x, _ = self.context_LSTM(x)
        return x


class ContextualEmbeddingLayerPos(nn.Module):
    """
    In: ((N, seq_len, emb_dim), max_len_context)
    Out: (N, seq_len, lstm_hidden_dim)
    """
    def __init__(self, config):
        super(ContextualEmbeddingLayerPos, self).__init__()

        self.context_LSTM = nn.LSTM(input_size=config.POS_EMBED_DIM,
                                 hidden_size=config.LSTM_HIDDEN_DIM,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=config.LSTM_DROP_RATE)
        self.dropout = nn.Dropout(config.LSTM_DROP_RATE)

    def forward(self, x):
        x = self.dropout(x)
        x, _ = self.context_LSTM(x)
        return x


# Attention Flow Layer
# ------------------------------------------------------------
class AttentionFlowLayer(nn.Module):
    """
    In:
        emb_context: (N, seq_len_c, lstm_hidden_dim)
        emb_query: (N, seq_len_q, lstm_hidden_dim)
    Out:
        G: (N, seq_len_c, 8*lstm_hidden_dim)

    """
    def __init__(self, config):
        super(AttentionFlowLayer, self).__init__()

        self.config = config

        self.W = nn.Linear(6*config.LSTM_HIDDEN_DIM, 1, bias=False)

    def forward(self, emb_context, emb_query):
        max_len_context, max_len_query = emb_context.size(1), emb_query.size(1)

        shape = (emb_context.shape[0], max_len_context,
                 max_len_query, self.config.LSTM_HIDDEN_DIM*2)  # (N, T, J, 2*h_dim)

        # construct similarity matrix
        emb_context_ex = emb_context.unsqueeze(2)           # (N, T, 1, 2*h_dim)
        emb_context_ex = emb_context_ex.expand(shape)  # (N, T, J, 2*h_dim)
        emb_query_ex = emb_query.unsqueeze(1)               # (N, 1, J, 2*h_dim)
        emb_query_ex = emb_query_ex.expand(shape)      # (N, T, J, 2*h_dim)

        cat_data = torch.cat((emb_context_ex,
                              emb_query_ex,
                              torch.mul(emb_context_ex, emb_query_ex)), 3)  # (N, T, J, 6*h_dim), [h;u;h◦u]
        S = self.W(cat_data).view(emb_context.shape[0],
                                  max_len_context,
                                  max_len_query)  # (N, T, J)

        # Context2Query attention
        c2q_attn = torch.bmm(F.softmax(S, dim=-1), emb_query)  # (N, T, h_dim) = bmm( (N, T, J), (N, J, h_dim) )

        # Query2Context attention
        b = F.softmax(torch.max(S, 2)[0], dim=-1)  # (N, T)
        q2c_attn = torch.bmm(b.unsqueeze(1), emb_context)  # (N, 1, 2d) = bmm( (N, 1, T), (N, T, h_dim) )
        q2c_attn = q2c_attn.repeat(1, max_len_context, 1)  # (N, T, h_dim), tiled T times

        # combined attention: query aware representation of each context word
        G = torch.cat((emb_context,
                       c2q_attn,
                       emb_context.mul(c2q_attn),
                       emb_context.mul(q2c_attn)),
                      2)  # (N, T, 4*h_dim)
        if self.config.OUTPUT_ATTN:
            return G, S

        return G


# Modeling Layer
# ------------------------------------------------------------
class ModelingLayer(nn.Module):
    """
    In:  (N, max_len_context, 8*h_dim)
    Out: (N, max_len_context, h_dim)
    """
    def __init__(self, config):
        super(ModelingLayer, self).__init__()

        self.modeling_LSTM1 = nn.LSTM(input_size=config.LSTM_HIDDEN_DIM * 8,
                                   hidden_size=config.LSTM_HIDDEN_DIM,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=config.LSTM_DROP_RATE)
        self.dropout = nn.Dropout(config.LSTM_DROP_RATE)

    def forward(self, x, seq_context_lens):
        x = self.dropout(x)
        x, _ = self.modeling_LSTM1(x)
        return x





