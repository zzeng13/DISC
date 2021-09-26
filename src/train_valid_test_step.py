#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training/validation/inference step for DISC model.
"""

import torch
from tqdm import tqdm
from src.utils.eval_util import *

__author__ = "Ziheng Zeng"
__email__ = "zzeng13@illinois.edu"
__status__ = "Prototype"


def train_step(model, optimizer, data_handler, criterion, epoch, writer):
    """Train model for one epoch"""
    model.train()
    # performance recorders
    loss_epoch = AverageMeter()
    seq_acc_epoch = AverageMeter()

    # train data for a single epoch
    bbar = tqdm(enumerate(data_handler.trainset_generator), ncols=100, leave=False)
    for idx, data in bbar:
        torch.cuda.empty_cache()
        batch_size = data['xs_bert'].shape[0]

        # model forward pass to compute the node embeddings
        ys_ = model(data)
        loss = criterion(ys_.reshape(-1, data_handler.config.TGT_VOCAB_SIZE), data['ys'].reshape(-1))

        # compute negative sampling loss and update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log eval metrics
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)

        # eval results
        xs = data['xs_bert'].cpu().detach().numpy()  # batch_size, max_xs_seq_len
        ys = data['ys'].cpu().detach().numpy()  # batch_size, max_ys_seq_len
        ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        xs, ys, ys_ = post_process_eval(xs, ys, ys_, data_handler.config)
        seq_acc = get_seq_acc(ys, ys_)
        seq_acc_epoch.update(seq_acc, 1)
        # random sample to show
        if data_handler.config.VERBOSE and idx % data_handler.config.DISPLAY_FREQ == 0:
            src, tar, pred = rand_sample(xs, ys, ys_)
            bbar.set_description("Phase: [Train] | Batch: {}/{} | Train Loss: {:.3f} | Seq Acc: {:.3f}\n src: {}\n tgt: {}\n pred: {}\n".format(idx,
                                                                                             data_handler.config.num_batch_train,
                                                                                             loss, seq_acc,
                                                                                             src, tar, pred))
        # set display bar
        else:
            bbar.set_description("Phase: [Train] | Batch: {}/{} | Train Loss: {:.5f} | Seq Acc: {:.3f}".format(idx,
                                                                                             data_handler.config.num_batch_train,
                                                                                             loss, seq_acc))
        if idx % data_handler.config.SAVE_FREQ == 0:
            if data_handler.config.USE_TENSORBOARD:
                writer.add_scalar('train_loss', loss_epoch.avg, epoch*data_handler.config.num_batch_train+idx)
                writer.add_scalar('train_seq_acc', seq_acc, epoch*data_handler.config.num_batch_train+idx)

    return loss_epoch.avg, seq_acc_epoch.avg


def valid_step(model, data_handler, criterion):
    """Valid model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    loss_epoch = AverageMeter()
    seq_acc_epoch = AverageMeter()

    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.validset_generator), ncols=100, leave=False)
    for idx, data in bbar:
        torch.cuda.empty_cache()
        batch_size = data['xs_bert'].shape[0]

        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            ys_ = model(data)
            loss = criterion(ys_.reshape(-1, data_handler.config.TGT_VOCAB_SIZE), data['ys'].reshape(-1))

        # log eval metrics
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)

        # eval results
        xs = data['xs_bert'].cpu().detach().numpy()  # batch_size, max_xs_seq_len
        ys = data['ys'].cpu().detach().numpy()  # batch_size, max_ys_seq_len
        ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        xs, ys, ys_ = post_process_eval(xs, ys, ys_, data_handler.config)
        seq_acc = get_seq_acc(ys, ys_)
        seq_acc_epoch.update(seq_acc, batch_size)

        # random sample to show
        if data_handler.config.VERBOSE:
            src, tar, pred = rand_sample(xs, ys, ys_)
            bbar.set_description("Phase: [Valid] | Batch: {}/{} | Valid Loss: {:.3f} | Seq Acc: {:.3f}\n src: {}\n tgt: {}\n pred: {}\n".format(idx,
                                                                                             data_handler.config.num_batch_valid,
                                                                                             loss,
                                                                                             seq_acc,
                                                                                             src, tar, pred))

        # set display bar
        else:
            bbar.set_description("Phase: [Valid] | Batch: {}/{} | Valid Loss: {:.3f} | Seq Acc: {:.3f}".format(idx,
                                                                                          data_handler.config.num_batch_valid,
                                                                                          loss,
                                                                                          seq_acc))
    return loss_epoch.avg, seq_acc_epoch.avg
