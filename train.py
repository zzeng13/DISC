#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training script for DISC model.
"""
from datetime import datetime
from tqdm import trange
from tensorboardX import SummaryWriter
from src.utils.data_util import *
from src.utils.model_util import *
from src.train_valid_test_step import *
from config import Config
# handle multi-processing for data loader
from torch.multiprocessing import set_start_method
from src.model.read_comp_triflow import ReadingComprehensionDetector as DetectorMdl


__author__ = "Ziheng Zeng"
__email__ = "zzeng13@illinois.edu"
__status__ = "Prototype"

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    """
    The training script for the DISC model.
    """
    # Initialize a data loader
    # ---------------------------------------------------------------------------------
    data_handler = DataHandler()

    # Manage and initialize model
    # ---------------------------------------------------------------------------------
    # Initialize model
    save_path = Config.PATH_TO_CHECKPOINT
    detector_model, optimizer, epoch_start = load_init_model(DetectorMdl, data_handler.config)
    # Freeze the pre-trained embedding layer
    for param in detector_model.bert_embedding_layer.embedding_model.base_model.parameters():
        param.requires_grad = False
    # Add the loss function
    criterion = torch.nn.NLLLoss(ignore_index=data_handler.config.PAD_IDX)
    # Add Tensorboard writer
    writer = None
    if Config.USE_TENSORBOARD:
        writer = SummaryWriter(log_dir='./runs/{}_{}'.format(Config.MODEL_NAME, datetime.today().strftime('%Y-%m-%d')))
    # Book-keeping info
    best_valid_acc = float('-inf')

    # Train model
    # ---------------------------------------------------------------------------------
    ebar = trange(epoch_start, Config.NUM_EPOCHS, desc='EPOCH', ncols=130, leave=True)
    set_seed(Config.SEED)
    for epoch in ebar:
        # Training
        _, _ = train_step(detector_model, optimizer, data_handler, criterion, epoch, writer)

        # Validation
        if epoch % Config.VALID_FREQ == 0:
            valid_loss, valid_seq_acc = valid_step(detector_model, data_handler, criterion)
            if best_valid_acc < valid_seq_acc:
                best_valid_acc = valid_seq_acc
                # save the best model seen so far
                save_model(save_path.format(Config.MODEL_NAME, 'best'), detector_model, optimizer, epoch)
            if Config.USE_TENSORBOARD:
                writer.add_scalar('valid_loss', valid_loss, epoch)
                writer.add_scalar('valid_seq_acc', valid_seq_acc, epoch)

        # save the latest model
        save_model(save_path.format(Config.MODEL_NAME, 'latest'), detector_model, optimizer, epoch)

    return


if __name__ == '__main__':
    train()


