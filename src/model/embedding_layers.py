import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, RobertaModel


class EmbeddingGeneratorGLOVE(nn.Module):
    def __init__(self, config, path):
        super(EmbeddingGeneratorGLOVE, self).__init__()
        self.config = config
        print('Loading Pre-trained Glove Embeddings...')
        embed_weights = np.load(path)
        vocab_size, dim = embed_weights.shape
        embed_weights = torch.FloatTensor(embed_weights)
        self.embedding_model = nn.Embedding(vocab_size, dim, padding_idx=config.PAD_IDX)
        self.embedding_model.weight = nn.Parameter(embed_weights)

    def forward(self, xs):
        # [batch_size, max_seq_len, hidden_dim]
        xs = self.embedding_model(xs)
        return xs


class EembeddingGeneratorBERT(nn.Module):
    """
    Pretrained Language Model - BERT
    """
    def __init__(self, config):
        super(EembeddingGeneratorBERT, self).__init__()
        self.embedding_model = BertModel.from_pretrained(
            config.PRETRAINED_BERT_NAME,
            return_dict=True
        )
        self.embedding_model.to(config.DEVICE)

    def forward(self, xs, attn_mask):
        xs = self.embedding_model(xs, attention_mask=attn_mask)
        # [batch_size, max_seq_len, hidden_dim]
        xs = xs.last_hidden_state  # extract the last hidden layer
        return xs


class EembeddingGeneratorRoBERTa(nn.Module):
    """
    Pretrained Language Model - RoBERTa
    """
    def __init__(self, config):
        super(EembeddingGeneratorRoBERTa, self).__init__()
        self.embedding_model = RobertaModel.from_pretrained(
            config.PRETRAINED_ROBERTA_NAME,
            return_dict=True
        )
        self.embedding_model.to(config.DEVICE)

    def forward(self, xs, attn_mask):
        xs = self.embedding_model(xs, attention_mask=attn_mask)
        # [batch_size, max_seq_len, hidden_dim]
        xs = xs.last_hidden_state  # extract the last hidden layer
        return xs


class CharacterEmbedding(nn.Module):
    '''
     In : (N, sentence_len, word_len)
     Out: (N, sentence_len, c_embd_size)

     Reference: https://github.com/jojonki/BiDAF/blob/master/layers/char_embedding.py
     '''
    def __init__(self, config):
        super(CharacterEmbedding, self).__init__()
        self.config = config
        self.embd_size = config.CHAR_EMBED_DIM
        self.embedding = nn.Embedding(config.CHAR_VOCAB_SIZE, config.CHAR_EMBED_DIM, padding_idx=config.PAD_IDX)
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv = nn.ModuleList([nn.Conv2d(1, config.CHAR_EMBED_CNN_NUM_OUT_CHANNELS,
                                             (f[0], f[1])) for f in config.CHAR_EMBED_CHAR_FILTERS])
        self.dropout = nn.Dropout(config.CHAR_EMBED_DROPOUT_RATE)

    def forward(self, x):
        # x: (N, seq_len, word_len)
        input_shape = x.size()
        # bs = x.size(0)
        # seq_len = x.size(1)
        word_len = x.size(2)
        x = x.view(-1, word_len) # (N*seq_len, word_len)
        x = self.embedding(x) # (N*seq_len, word_len, c_embd_size)
        x = x.view(*input_shape, -1) # (N, seq_len, word_len, c_embd_size)
        x = x.sum(2) # (N, seq_len, c_embd_size)

        # CNN
        x = x.unsqueeze(1) # (N, Cin, seq_len, c_embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (N,Cin, Hin, Win )
        #    Output: (N,Cout,Hout,Wout)
        x = [F.relu(conv(x)) for conv in self.conv] # (N, Cout, seq_len, c_embd_size-filter_w+1). stride == 1
        # [(N,Cout,Hout,Wout) -> [(N,Cout,Hout*Wout)] * len(filter_heights)
        # [(N, seq_len, c_embd_size-filter_w+1, Cout)] * len(filter_heights)
        x = [xx.view((xx.size(0), xx.size(2), xx.size(3), xx.size(1))) for xx in x]
        # maxpool like
        # [(N, seq_len, Cout)] * len(filter_heights)
        x = [torch.sum(xx, 2) for xx in x]
        # (N, seq_len, Cout==word_embd_size)
        x = torch.cat(x, 1)
        x = self.dropout(x)

        return x


class EembeddingGeneratorPOS(nn.Module):
    def __init__(self, config):
        super(EembeddingGeneratorPOS, self).__init__()
        self.embedding_model = nn.Embedding(config.POS_VOCAB_SIZE, config.POS_EMBED_DIM, padding_idx=config.PAD_IDX)
        self.embedding_model.to(config.DEVICE)

    def forward(self, xs):
        xs = self.embedding_model(xs)
        # [batch_size, max_seq_len, hidden_dim]
        return xs