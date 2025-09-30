import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def load_weight(path, model, start_keys=['two_stream_net', 'fusion_net'], freeze=True):
    model_dict = model.state_dict()
    save_dict = torch.load(path)['state_dict']

    update_dict = {
        key: save_dict[key] for key in save_dict.keys()
        if any([key.startswith(start_key) for start_key in start_keys])
    }

    model_dict.update(update_dict)
    model.load_state_dict(model_dict)

    if freeze:
        update_keys = set(update_dict.keys())
        for name, param in model.named_parameters():
            if name in update_keys:
                param.requires_grad = False 

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('train_loader', module_data)
    valid_data_loader = config.init_obj('val_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)



    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    print(data_loader)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      lr_scheduler=lr_scheduler,
                      valid_data_loader=valid_data_loader)
    
    best_model_path  = trainer.train()
    if best_model_path:
        print(best_model_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
