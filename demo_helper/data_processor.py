from itertools import islice
import torch
from torch.utils import data as torch_data
from torch.nn.utils.rnn import pad_sequence
from src.utils.file_util import load_json_file
from src.model.embedding_layers import EmbeddingGeneratorGLOVE
from transformers import BertTokenizerFast
from nltk.tokenize import word_tokenize
import nltk


class DataHandler(object):
    def __init__(self, config):
        super(DataHandler, self).__init__()
        self.config = config
        self.update_config()
        self.tokenizer = BertTokenizerFast.from_pretrained(config.PRETRAINED_BERT_NAME)

    def update_config(self):
        # read vocabs:
        path_to_data_files = load_json_file(self.config.PATH_TO_META_DATA)
        self.target_vocab = load_json_file(path_to_data_files['path_to_target_vocab'])
        self.char_vocab = load_json_file(path_to_data_files['path_to_char_vocab'])
        self.pos_vocab = load_json_file((path_to_data_files['path_to_pos_vocab']))
        self.glove_vocab = load_json_file((path_to_data_files['path_to_glove_vocab']))


        # self.target_vocab = {v:k for k, v in self.target_vocab.items()}
        self.char_vocab = {v: k for k, v in self.char_vocab.items()}
        self.pos_vocab = {v: k for k, v in self.pos_vocab.items()}
        self.glove_vocab = {v: k for k, v in self.glove_vocab.items()}


        self.config.TGT_VOCAB_SIZE = len(self.target_vocab.keys())
        self.config.NUM_CLASSES = self.config.TGT_VOCAB_SIZE
        self.config.CHAR_VOCAB_SIZE = len(self.char_vocab.keys())
        self.config.POS_VOCAB_SIZE = len(self.pos_vocab.keys())
        self.config.START_IDX = self.target_vocab[self.config.START_SYMBOL]
        self.config.END_IDX = self.target_vocab[self.config.END_SYMBOL]
        self.config.PAD_IDX = self.target_vocab[self.config.PAD_SYMBOL]
        self.glove_embedding_layer = EmbeddingGeneratorGLOVE(self.config, path_to_data_files['path_to_glove_embed'])

    def get_bert_tokens(self, sentences):
        bert_tkn = self.tokenizer.batch_encode_plus(sentences,
                                                    truncation=True, max_length=self.config.MAX_SEQ_LEN, padding=True,
                                                    return_tensors="pt")
        return bert_tkn

    def get_glove_and_pos_and_chartokens(self, sentences):
        pos_tokens, glove_tokens, char_tokens = [], [], []
        for sent in sentences:
            source_sentence_glove_tknz = word_tokenize(sent)
            tags_tokens = nltk.pos_tag(source_sentence_glove_tknz)
            source_sentence_glove_tknz = [t[0] for t in tags_tokens]
            source_sentence_pos_taggs = [t[1] for t in tags_tokens]
            source_sentence_glove_words = ['<S>'] + source_sentence_glove_tknz + ['<E>']
            source_sentence_pos_taggs = ['<S>'] + source_sentence_pos_taggs + ['<E>']
            source_sentence_glove_tknz = [int(self.glove_vocab[t]) if t in self.glove_vocab else int(self.glove_vocab['<UNK>']) for t in source_sentence_glove_words]
            source_sentence_pos_taggs = [int(self.pos_vocab[t]) if t in self.pos_vocab else int(self.pos_vocab['<UNK>']) for t in source_sentence_pos_taggs]
            glove_tokens.append(torch.Tensor(source_sentence_glove_tknz))
            pos_tokens.append(torch.Tensor(source_sentence_pos_taggs))
            # ----------------
            source_sentence_char_tknz = []
            for word in source_sentence_glove_words:
                if word in ['<S>', '<E>', '<UNK>']:
                    source_sentence_char_tknz.append([int(self.char_vocab['<SPEC>'])])
                else:
                    source_sentence_char_tknz.append([int(self.char_vocab[w]) if w in self.char_vocab else int(self.char_vocab['<UNK>']) for w in list(word)])
            char_tokens.append(source_sentence_char_tknz)

        xs_glove = pad_sequence(glove_tokens, batch_first=True, padding_value=self.config.PAD_IDX)
        xs_pos = pad_sequence(pos_tokens, batch_first=True, padding_value=self.config.PAD_IDX)
        mask = xs_pos != self.config.PAD_IDX
        xs_glove_lens = mask.sum(-1)
        # ----------------
        xs_char = [torch.Tensor(item) for sublist in char_tokens for item in sublist]
        xs_char = pad_sequence(xs_char, batch_first=True, padding_value=self.config.PAD_IDX)
        xs_char = [xs_char[i] for i in range(xs_char.shape[0])]
        xs_char = iter(xs_char)
        xs_char = [list(islice(xs_char, elem))
                   for elem in xs_glove_lens.cpu().detach().numpy().tolist()]
        xs_char = [torch.vstack(seq) for seq in xs_char]
        xs_char = pad_sequence(xs_char, batch_first=True, padding_value=self.config.PAD_IDX)

        return xs_glove, xs_pos, xs_char, xs_glove_lens

    def prepare_input(self, sentences):
        """
        Given a list of sentences, process and prepare the input for the detector.
        """
        sentences = [apply_contraction_change(s) for s in sentences]
        bert_tkn = self.get_bert_tokens(sentences)
        xs_glove, xs_pos, xs_char, xs_glove_lens = self.get_glove_and_pos_and_chartokens(sentences)
        xs_glove = self.glove_embedding_layer(xs_glove.long())
        xs_bert = bert_tkn['input_ids']
        attn_mask = bert_tkn['attention_mask']
        xs_bert_lens = attn_mask.sum(-1)

        return {'xs_bert': xs_bert.long().to(self.config.DEVICE),
                'xs_glove': xs_glove.to(self.config.DEVICE),
                'xs_char': xs_char.long().to(self.config.DEVICE),
                'xs_pos': xs_pos.long().to(self.config.DEVICE),
                'xs_bert_lens': xs_bert_lens.long().to(self.config.DEVICE),
                'xs_bert_attn_mask': attn_mask.long().to(self.config.DEVICE),
                'xs_glove_lens': xs_glove_lens}


def apply_contraction_change(s):
    s = s.lower()
    s = s.replace(" n't", "n't")
    s = s.replace("\n", "")
    s = s.replace("‘", " ‘ ")
    s = s.replace("’", " ’ ")
    s = s.replace(",", " , ")
    s = s.replace(".", " . ")
    s = s.replace('?', ' ? ')
    s = s.replace('!', ' ! ')
    s = s.replace('-', ' - ')
    return s


