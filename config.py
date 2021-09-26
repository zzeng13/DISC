#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Experiment Configuration.
"""
from os.path import join, abspath, dirname
import torch

__author__ = "Ziheng Zeng"
__email__ = "zzeng13@illinois.edu"
__status__ = "Prototype"


class Config:
    ROOT = abspath(dirname(__file__))

    # Settings - (regularly changed)
    # ==============================================================================================================
    MODE = 'test'  # 'train' or 'test' # Must be set manually every time
    DATA_NAME = 'magpie'
    SPLIT = 'random'
    MODEL_TYPE = 'cross_attn-glove-char-pos-tri'
    MODEL_NAME = 'ReadComp_{}_{}_{}'.format(DATA_NAME, SPLIT, MODEL_TYPE)  # name the current training or testing model
    PATH_TO_META_DATA = './meta_data/meta_data_{}_{}.json'.format(DATA_NAME, SPLIT)  # file that records data paths

    SEED = 123
    OUTPUT_ATTN = False

    # Book-keeping
    USE_GPU = True
    CONTINUE_TRAIN = False
    USE_TENSORBOARD = True
    VERBOSE = False  # display sampled prediction results
    NUM_WORKER = 0

    # Checkpoint management
    PATH_TO_CHECKPOINT = join(ROOT, 'checkpoints/{}_{}.mdl')
    LOAD_CHECKPOINT_TYPE = 'best'  # 'latest' or 'best

    # ++++++++++++++++++++++++++++++++++++++++++ PARAMETERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Train Parameters
    # ==============================================================================================================
    NUM_EPOCHS = 600
    BATCH_SIZE = 56
    VALID_FREQ = 1  # number of epochs to run validation
    SAVE_FREQ = 10  # number of steps to save train performance
    DISPLAY_FREQ = 10  # number of steps to display train performance (only matters if VERBOSE==TRUE)
    LEARNING_RATE = 1e-4

    # Inference Parameters
    # ==============================================================================================================
    PATH_TO_SAVE_PERFORMANCE = join(ROOT, 'res/{}_inference_performance.json'.format(MODEL_NAME))
    PATH_TO_SAVE_RESULTS = join(ROOT, 'res/{}_{}_inference_results.json'.format(DATA_NAME, SPLIT))

    # Data Parameters
    # ==============================================================================================================
    MAX_SEQ_LEN = 512

    START_SYMBOL = '<s>'
    END_SYMBOL = '<e>'
    PAD_SYMBOL = '<PAD>'

    # Model Parameters
    # ==============================================================================================================
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")

    # Embeddings
    PRETRAINED_BERT_NAME = 'bert-base-uncased'
    PRETRAINED_BERT_EMBED_DIM = 768
    PRETRAINED_LM_EMBED_DIM = PRETRAINED_BERT_EMBED_DIM
    PRETRAINED_GLOVE_EMBED_DIM = 300
    CHAR_EMBED_DIM = 64
    CHAR_EMBED_DROPOUT_RATE = 0.2
    CHAR_EMBED_CNN_NUM_OUT_CHANNELS = 64
    CHAR_EMBED_CHAR_FILTERS = [[1, 5]]
    POS_EMBED_DIM = 64
    # fused embedding dimension
    EMBEDDING_DIM = PRETRAINED_GLOVE_EMBED_DIM + CHAR_EMBED_CNN_NUM_OUT_CHANNELS

    # Highway Network
    HIGHWAY_NUM_LAYERS = 2

    # LSTM
    LSTM_HIDDEN_DIM = 256
    LSTM_DROP_RATE = 0.3

    # Mode based parameters
    if MODE != 'train':
        LOAD_CHECKPOINT_TYPE = 'latest'
        CONTINUE_TRAIN = True
        USE_TENSORBOARD = False
        BATCH_SIZE = 1



