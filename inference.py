#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Inference script for DISC model.
"""
from tqdm import tqdm
from src.utils.data_util import *
from src.utils.model_util import *
from src.model.read_comp_triflow import ReadingComprehensionDetector as DetectorMdl

from src.utils.eval_util import *
# handle multi-processing for data loader
from torch.multiprocessing import set_start_method

__author__ = "Ziheng Z"
__email__ = "zzeng13@illinois.edu"
__status__ = "Prototype"

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def infer_step(model, data_handler):
    """Infer model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    seq_acc_epoch = AverageMeter()
    cls_tars, cls_preds = [], []
    ys_pred_correctness = []
    target_seqs = []
    predict_seqs = []
    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.testset_generator), ncols=100, leave=True, total=len(data_handler.testset_generator))
    for idx, data in bbar:
        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            ys_ = model(data)

        # eval results
        batch_size = data['xs_bert'].shape[0]
        xs = data['xs_bert'].cpu().detach().numpy()  # batch_size, max_xs_seq_len
        ys = data['ys'].cpu().detach().numpy()  # batch_size, max_ys_seq_len

        ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy()  # batch_size, max_ys_seq_len
        xs, ys, ys_ = post_process_eval(xs, ys, ys_, data_handler.config)
        target_seqs.append(ys)
        predict_seqs.append(ys_)
        seq_acc, correctness = get_seq_acc_test(ys, ys_)
        ys_pred_correctness += correctness
        cur_cls_tars, cur_cls_preds = get_cls_acc_test(ys, ys_)
        cls_tars += cur_cls_tars
        cls_preds += cur_cls_preds
        seq_acc_epoch.update(seq_acc, batch_size)

        # format data
        for i in range(len(ys_)):
            ys_[i] = [s.item() for s in ys_[i]]

        bbar.set_description("Phase: [Test] | Batch: {}/{} | Seq Acc: {:.3f}".format(idx,
                                                                                      data_handler.config.num_batch_test,
                                                                                    seq_acc))
    print(sum(ys_pred_correctness)/len(ys_pred_correctness))
    return target_seqs, predict_seqs, cls_tars, cls_preds, ys_pred_correctness


def inference():
    """
    The training script for the DISC model.
    """
    # Initialize a data loader
    # ---------------------------------------------------------------------------------
    data_handler = DataHandler()

    # Manage and initialize model
    # ---------------------------------------------------------------------------------
    # Initialize model
    detector_model, optimizer, epoch_start = load_init_model(DetectorMdl, data_handler.config)

    # Run model inference
    # ---------------------------------------------------------------------------------
    target_seqs, predict_seqs, cls_tars, cls_preds, ys_pred_correctness = infer_step(detector_model, data_handler)

    write_json_file(data_handler.config.PATH_TO_SAVE_RESULTS,
                    {'seq_tars': [[int(j) for j in i[0]] for i in target_seqs],
                     'seq_preds': [[int(j) for j in i[0]] for i in predict_seqs],
                     'cls_tars': [int(j) for j in cls_tars],
                     'cls_preds': [int(j) for j in cls_preds],
                     'seq_correctness': [int(j) for j in ys_pred_correctness]})


if __name__ == '__main__':
    inference()

