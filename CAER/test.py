import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('test_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=False)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)

            pixel_values_face = data['pixel_values_face'].to(device)
            pixel_values_context = data['pixel_values_context'].to(device)
            target = target.to(device)
            output_dict = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            pixel_values_context=pixel_values_context,
                            pixel_values_face=pixel_values_face
            )
            output = output_dict['cat_pred']

            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_percent = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_percent[i] = (cm[i] / row_sums[i]) * 100

    class_names = config.config.get('class_names', [str(i) for i in range(cm.shape[0])])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix (%)')
    
    output_dir = config.log_dir
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
