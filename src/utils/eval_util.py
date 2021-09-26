from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def rm_idx(seq, idx):
    return [i for i in seq if i != idx]


def post_process(srcs, tgts, preds, config):
    # remove pad idx
    srcs = [rm_idx(src, config.PAD_IDX) for src in srcs]
    tgts = [rm_idx(tgt, config.PAD_IDX) for tgt in tgts]
    preds = [rm_idx(pred, config.PAD_IDX) for pred in preds]

    # remove end idx
    end_indices = [p.index(config.END_IDX)+1 if config.END_IDX in p else len(p) for p in preds]
    preds = [p[:idx] for idx, p in zip(end_indices, preds)]
    return srcs, tgts, preds

def post_process_eval(srcs, tgts, preds, config):
    # remove pad idx
    srcs = [[p for p in src] for src in srcs]
    tgts = [rm_idx(tgt, config.PAD_IDX) for tgt in tgts]
    preds = [[p for p in pred] for pred in preds]

    # remove end idx
    end_indices = [len(p) for p in tgts]
    preds = [p[:idx] for idx, p in zip(end_indices, preds)]
    return srcs, tgts, preds


def rand_sample(srcs, tars, preds):
    src, tar, pred = random.choice([(src, tar, pred) for src, tar, pred in zip(srcs, tars, preds)])
    src = [str(s) for s in src]
    tar = [str(s) for s in tar]
    pred = [str(s) for s in pred]

    return ' '.join(src), ' '.join(tar), ' '.join(pred)


def check_seq(tar, pred):
    min_len = min([len(tar), len(pred)])
    if sum(np.equal(tar[:min_len], pred[:min_len])) == len(tar):
        return 1
    return 0


def check_cls(tar, pred):
    # TODO: replace the magic number here
    tar_cls = 1 if 4 in tar else 0
    pred_cls = 1 if 4 in pred else 0
    return tar_cls, pred_cls


def get_seq_acc(tars, preds):
    size = len(tars)
    a = 0
    for i in range(size):
        tar = tars[i]
        pred = preds[i]
        a += check_seq(tar, pred)

    return np.float32(a/size)


def get_seq_acc_test(tars, preds):
    size = len(tars)
    a = 0
    pred_seq = []
    for i in range(size):
        tar = tars[i]
        pred = preds[i]
        correctness = check_seq(tar, pred)
        a += correctness
        pred_seq.append(correctness)

    return np.float32(a/size), pred_seq


def get_cls_acc_test(tars, preds):
    size = len(tars)
    cur_tars, cur_preds = [], []
    for i in range(size):
        tar = tars[i]
        pred = preds[i]
        t, p = check_cls(tar, pred)
        cur_tars.append(t)
        cur_preds.append(p)

    return cur_tars, cur_preds


def get_seq_performance(tars, preds):
    size = len(tars)
    tars_tok_seq = []
    pred_tok_seq = []

    for i in range(size):
        tar = tars[i]
        pred = preds[i]
        tars_tok_seq += tar
        pred_tok_seq += pred
    # compute token level performance
    tars_tok_seq = [item for sublist in tars_tok_seq for item in sublist]
    pred_tok_seq = [item for sublist in pred_tok_seq for item in sublist]
    # TODO: remove magic numbers
    tars_tok_seq = [1 if t == 4 else 0 for t in tars_tok_seq]
    pred_tok_seq = [1 if t == 4 else 0 for t in pred_tok_seq]
    tok_acc = accuracy_score(tars_tok_seq, pred_tok_seq)
    tok_pre = precision_score(tars_tok_seq, pred_tok_seq)
    tok_rec = recall_score(tars_tok_seq, pred_tok_seq)
    tok_f1 = f1_score(tars_tok_seq, pred_tok_seq)

    return {
        'tok_acc': tok_acc,
        'tok_pre': tok_pre,
        'tok_rec': tok_rec,
        'tok_f1': tok_f1
    }


def get_cls_performance(tars, preds):
    tok_acc = accuracy_score(tars, preds)
    tok_pre = precision_score(tars, preds)
    tok_rec = recall_score(tars, preds)
    tok_f1 = f1_score(tars, preds)

    return {
        'cls_acc': tok_acc,
        'cls_pre': tok_pre,
        'cls_rec': tok_rec,
        'cls_f1': tok_f1
    }