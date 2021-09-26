from src.model.attention_flow import *
from src.model.highway_network import HighwayNetwork
from src.model.embedding_layers import *


class ReadingComprehensionDetector(nn.Module):
    def __init__(self, config):
        super(ReadingComprehensionDetector, self).__init__()
        self.config = config
        # Embedding layers
        # ------------------------------------------------------
        # BERT embedding layer
        self.bert_embedding_layer = EembeddingGeneratorBERT(config)
        self.bert_embedding_transform_layer = nn.Linear(config.PRETRAINED_BERT_EMBED_DIM,
                                                        config.PRETRAINED_GLOVE_EMBED_DIM + config.CHAR_EMBED_DIM )
        # Glove embedding layer
        # self.glove_embedding_layer = EmbeddingGeneratorGLOVE(config)
        # character embedding layer
        self.char_embedding_layer = CharacterEmbedding(config)
        # POS embedding layer
        self.pos_embedding_layer = EembeddingGeneratorPOS(config)
        # Highway network
        self.highway_network = HighwayNetwork(config)

        # # Contextual Embedding layer
        self.contextual_embedding_layer = ContextualEmbeddingLayer(config)
        self.contextual_embedding_layer_pos = ContextualEmbeddingLayerPos(config)

        # Attention flow layer
        # ------------------------------------------------------
        self.attention_flow_layer_0 = AttentionFlowLayer(config)
        self.attention_flow_layer_1 = AttentionFlowLayer(config)

        # Modeling layer
        # ------------------------------------------------------
        self.modeling_layer_0 = ModelingLayer(config)
        self.modeling_layer_1 = ModelingLayer(config)

        # Output layer
        # ------------------------------------------------------
        self.output_layer = nn.Linear(config.LSTM_HIDDEN_DIM * 2, config.NUM_CLASSES)

    def forward(self, data):
        # shape: [batch_size, max_bert_seq_len]
        bert_token_seq = data['xs_bert']
        # shape: [batch_size, max_glove_seq_len, glove_embed_dim]
        glove_token_seq = data['xs_glove']
        # shape: [batch_size, max_glove_seq_len]
        pos_token_seq = data['xs_pos']
        # shape: [batch_size, max_glove_seq_len, max_word_len]
        char_seq = data['xs_char']
        # shape: [batch_size]
        seq_context_lens = data['xs_bert_lens']
        # shape: [batch_size]
        seq_query_lens = data['xs_glove_lens']
        bert_attn_mask = data['xs_bert_attn_mask']


        # 1. Embedding Layers
        # ------------------------------------------------------------------
        # shape: [batch_size, max_bert_seq_len, glove_embed_dim + char_embed_dim + pos_embed_dim]
        context_seq = self.bert_embedding_layer(bert_token_seq, bert_attn_mask)
        context_seq = self.bert_embedding_transform_layer(context_seq)
        del bert_token_seq
        # shape: [batch_size, max_glove_seq_len, glove_embed_dim]
        # glove_token_seq = self.glove_embedding_layer(glove_token_seq)
        # shape: [batch_size, max_glove_seq_len, pos_embed_dim]
        pos_token_seq = self.pos_embedding_layer(pos_token_seq)
        # shape: [batch_size, max_glove_seq_len, char_embed_dim]
        char_seq = self.char_embedding_layer(char_seq)
        # shape: [batch_size, max_glove_seq_len, glove_embed_dim + char_embed_dim + pos_embed_dim]
        query_seq = torch.cat((char_seq, glove_token_seq), -1)
        # shape: [batch_size, max_glove_seq_len, glove_embed_dim + char_embed_dim + pos_embed_dim]
        query_seq = self.highway_network(query_seq)
        del glove_token_seq, char_seq

        # 2. Contextual Embedding Layer
        # -------------------------------------------------------------------
        # shape: [batch_size, max_bert_seq_len, 2 * lstm_hidden_dim]
        context_seq = self.contextual_embedding_layer(context_seq)  # (N, T, 2d)
        # shape: [batch_size, max_glove_seq_len, 2 * lstm_hidden_dim]
        query_seq = self.contextual_embedding_layer(query_seq)  # (N, J, 2d)
        # shape: [batch_size, max_glove_seq_len, 2 * lstm_hidden_dim]
        pos_token_seq = self.contextual_embedding_layer_pos(pos_token_seq)
        # shape: [batch_size, max_bert_seq_len, 8 * lstm_hidden_dim ]
        if self.config.OUTPUT_ATTN:
            G_0, S_0 = self.attention_flow_layer_0(query_seq, pos_token_seq)
        else:
            G_0 = self.attention_flow_layer_0(query_seq, pos_token_seq)

        # shape: [batch_size, max_bert_seq_len, lstm_hidden_dim * 2]
        M_0 = self.modeling_layer_0(G_0, seq_query_lens)

        # 3. Attention Flow Layer
        # -------------------------------------------------------------------
        # shape: [batch_size, max_bert_seq_len, 8 * lstm_hidden_dim ]
        # G_1 = self.attention_flow_layer_1(context_seq, M_0)  # (N, T, 8d) BiDAF paper
        if self.config.OUTPUT_ATTN:
            G_1, S_1 = self.attention_flow_layer_1(context_seq, M_0)
        else:
            G_1 = self.attention_flow_layer_1(context_seq, M_0)  # (N, T, 8d) BiDAF paper

        # 4. Modeling Layer
        # -------------------------------------------------------------------
        # shape: [batch_size, max_bert_seq_len, lstm_hidden_dim * 2]
        M_1 = self.modeling_layer_1(G_1, seq_context_lens)  # M: (N, T, 2d)

        # 5. Final Linear Layer (Seq2seq Bilstm with attention)
        # -------------------------------------------------------------------
        # shape: [batch_size, max_bert_seq_len, num_classes]
        out = self.output_layer(M_1)
        if self.config.OUTPUT_ATTN:
            return F.log_softmax(out, dim=-1), S_1

        return F.log_softmax(out, dim=-1)

