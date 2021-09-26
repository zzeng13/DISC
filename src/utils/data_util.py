#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Data util for Reading Comprehension-based subsequence detector model. Data packing and processing.
"""
from itertools import islice
import torch
from torch.utils import data as torch_data
from torch.nn.utils.rnn import pad_sequence
from src.utils.file_util import *
from config import Config

__author__ = "Ziheng Zeng"
__email__ = "zzeng13@illinois.edu"
__status__ = "Prototype"


# Data handler for training and validation data
class Dataset(torch_data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, xs):
        super(Dataset, self).__init__()
        self.xs = xs
        self.num_total_seqs = len(self.xs)

    def __len__(self):
        return self.num_total_seqs

    def __getitem__(self, index):
        return self.xs[index]


class DataHandler(object):

    def __init__(self):
        super(DataHandler, self).__init__()
        self.config = Config()
        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        path_to_data_files = load_json_file(self.config.PATH_TO_META_DATA)
        self.raw_data = load_json_file(path_to_data_files['path_to_raw_data'])
        self.target_vocab = load_json_file(path_to_data_files['path_to_target_vocab'])
        self.char_vocab = load_json_file(path_to_data_files['path_to_char_vocab'])
        self.pos_vocab = load_json_file((path_to_data_files['path_to_pos_vocab']))
        # self.config.PATH_TO_GLOVE_EMBEDDINGS = path_to_data_files['path_to_glove_embed']
        self.glove_feat_matrix = np.load(path_to_data_files['path_to_glove_embed'])

    def init_generators(self):
        if self.config.MODE == 'train':

            self.train_dataset = Dataset(self.raw_data['train'])
            self.trainset_generator = torch_data.DataLoader(self.train_dataset,
                                                            batch_size=self.config.BATCH_SIZE,
                                                            collate_fn=self.collate_fn,
                                                            shuffle=True,
                                                            num_workers=self.config.NUM_WORKER,
                                                            drop_last=True)
            # data loader for validset
            self.valid_dataset = Dataset(self.raw_data['valid'])
            self.validset_generator = torch_data.DataLoader(self.valid_dataset,
                                                            batch_size=self.config.BATCH_SIZE,
                                                            collate_fn=self.collate_fn,
                                                            shuffle=False,
                                                            num_workers=self.config.NUM_WORKER,
                                                            drop_last=False)
        else:
            self.test_dataset = Dataset(self.raw_data['valid'])
            self.testset_generator = torch_data.DataLoader(self.test_dataset,
                                                            batch_size=self.config.BATCH_SIZE,
                                                            collate_fn=self.collate_fn,
                                                            shuffle=False,
                                                            num_workers=self.config.NUM_WORKER,
                                                            drop_last=False)

    def update_config(self):
        def get_batch_size(dataset_size):
            if dataset_size % self.config.BATCH_SIZE == 0:
                return dataset_size // self.config.BATCH_SIZE
            else:
                return dataset_size // self.config.BATCH_SIZE + 1

        # training parameters
        if self.config.MODE == 'train':
            self.config.train_size = len(self.train_dataset)
            self.config.valid_size = len(self.valid_dataset)
            print('Training dataset size: {}'.format(self.config.train_size))
            print('Validation dataset size: {}'.format(self.config.valid_size))
            self.config.num_batch_train = get_batch_size(self.config.train_size)
            self.config.num_batch_valid = get_batch_size(self.config.valid_size)
        else:
            self.config.test_size = len(self.test_dataset)
            print('Testing dataset size: {}'.format(self.config.test_size))
            self.config.num_batch_test = get_batch_size(self.config.test_size)
        # data parameters
        # self.config.SRC_VOCAB_SIZE = len(self.vocab.keys())
        self.config.TGT_VOCAB_SIZE = len(self.target_vocab.keys())
        self.config.NUM_CLASSES = self.config.TGT_VOCAB_SIZE
        self.config.CHAR_VOCAB_SIZE = len(self.char_vocab.keys())
        self.config.POS_VOCAB_SIZE = len(self.pos_vocab.keys())
        self.config.START_IDX = self.target_vocab[self.config.START_SYMBOL]
        self.config.END_IDX = self.target_vocab[self.config.END_SYMBOL]
        self.config.PAD_IDX = self.target_vocab[self.config.PAD_SYMBOL]

    def collate_fn(self, data):
        # 1. Unpack the sequences
        xs_bert, xs_glove, xs_char, xs_pos, ys, labels = zip(*data)
        xs_bert, xs_glove, xs_char, xs_pos, ys, labels = list(xs_bert), list(xs_glove), list(xs_char), list(xs_pos), list(ys), list(labels)
        batch_size = len(labels)

        # Data pre-processing
        # 2. convert lists to tensors
        xs_glove = [self.glove_feat_matrix[xs_glove[i]] for i in range(batch_size)]
        xs_glove = [torch.Tensor(seq) for seq in xs_glove]
        xs_glove = pad_sequence(xs_glove, batch_first=True, padding_value=self.config.PAD_IDX)

        xs_bert = [torch.Tensor(seq) for seq in xs_bert]
        xs_pos = [torch.Tensor(seq) for seq in xs_pos]
        xs_char = [torch.Tensor(item) for sublist in xs_char for item in sublist]

        ys = [torch.Tensor(seq) for seq in ys]

        # 3. create pad sequence
        xs_bert = pad_sequence(xs_bert, batch_first=True, padding_value=self.config.PAD_IDX)
        xs_char = pad_sequence(xs_char, batch_first=True, padding_value=self.config.PAD_IDX)
        xs_char = [xs_char[i] for i in range(xs_char.shape[0])]
        xs_pos = pad_sequence(xs_pos, batch_first=True, padding_value=self.config.PAD_IDX)

        ys = pad_sequence(ys, batch_first=True, padding_value=self.config.PAD_IDX)

        # 4. record source seq lengths
        attn_mask = xs_bert != self.config.PAD_IDX
        xs_bert_lens = attn_mask.sum(-1)
        mask = xs_pos != self.config.PAD_IDX
        xs_glove_lens = mask.sum(-1)
        xs_char = iter(xs_char)
        xs_char = [list(islice(xs_char, elem))
                   for elem in xs_glove_lens.cpu().detach().numpy().tolist()]
        xs_char = [torch.vstack(seq) for seq in xs_char]
        xs_char = pad_sequence(xs_char, batch_first=True, padding_value=self.config.PAD_IDX)

        return {'xs_bert': xs_bert.long().to(self.config.DEVICE),
                'xs_glove': xs_glove.to(self.config.DEVICE),
                'xs_char': xs_char.long().to(self.config.DEVICE),
                'xs_pos': xs_pos.long().to(self.config.DEVICE),
                'ys': ys.long().to(self.config.DEVICE),
                'xs_bert_lens': xs_bert_lens.long().to(self.config.DEVICE),
                'xs_bert_attn_mask': attn_mask.long().to(self.config.DEVICE),
                'xs_glove_lens': xs_glove_lens,
                'labels': labels}

