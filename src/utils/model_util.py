import os
import torch


# Manage model and checkpoint loading
def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar', device='cpu'):
    '''
    Load a checkpoint.

    # Parameters

    :param model: pytorch model
    :param optimizer: pytorch optimizer
    :param filename: (str) path to saved checkpoint

    # Returns

    :return model: loaded model
    :return optimizer: optimizer with right parameters
    :return start_epoch: (int) epoch number load from model

    '''
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.

    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        if device == 'cpu':
            checkpoint = torch.load(filename, map_location=lambda storage, location: storage)
        else:
            checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'"
              .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def load_init_model(model_module, config):
    """
    Initialize a model and load a checkpoint if so desired (if the checkpoint is available.)

    # Parameters

    :param model_module: the class of the model.
    :param config: config class that contains all the parameters

    # Returns

    :return model: initialized model (loaded checkpoint)
    :return optimizer: initialized optimizer
    :return epoch_start: the starting epoch to continue the training
    """

    epoch_start = 0

    model = model_module(config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    save_path = config.PATH_TO_CHECKPOINT.format(config.MODEL_NAME, config.LOAD_CHECKPOINT_TYPE)
    if os.path.exists(save_path) and config.CONTINUE_TRAIN:
        print('Loading model from {}'.format(save_path))
        model, optimizer, epoch_start = load_checkpoint(model, optimizer, save_path, config.DEVICE)
    else:
        print('=> No checkpoint found! Train from scratch!')
    model.eval()

    return model, optimizer, epoch_start


def load_model_from_checkpoint(model_module, config):
    """
    Initialize a model and load a checkpoint.

    # Parameters

    :param model_module: the class of the model.
    :param config: config class that contains all the parameters

    # Returns

    :return model: initialized model (loaded checkpoint)
    :return optimizer: initialized optimizer
    :return epoch_start: the starting epoch to continue the training
    """

    model = model_module(config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    save_path = config.PATH_TO_CHECKPOINT.format(config.MODEL_NAME, config.LOAD_CHECKPOINT_TYPE)
    if os.path.exists(save_path):
        print('Loading model from {}'.format(save_path))
        model, _, _ = load_checkpoint(model, optimizer, save_path, config.DEVICE)
    else:
        raise NotImplementedError('=> No checkpoint found!')
    model.eval()

    return model


def save_model(save_path, model, optimizer, epoch):
    """
    Save the model, loss, and optimizer to a checkpoint
    """
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    # save the best model seen so far
    torch.save(state, save_path)


def count_parameters(model):
    # get total size of trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

