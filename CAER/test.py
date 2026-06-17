import argparse
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            if batch is None:
                continue
            data, target = batch

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)

            pixel_values_face = data['pixel_values_face'].to(device)
            pixel_values_context = data['pixel_values_context'].to(device)
            target = target.to(device)
            output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            pixel_values_context=pixel_values_context,
                            pixel_values_face=pixel_values_face
            )

            # Lấy tensor logit (do model trả về dictionary)
            output_logits = output['cat_pred'] if isinstance(output, dict) else output
            
            # Lưu lại dự đoán và nhãn thực tế
            preds = torch.argmax(output_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # computing loss, metrics on test set
            loss = loss_fn(output_logits, target)
            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output_logits, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)
    
    # In Confusion Matrix và Classification Report
    cm = confusion_matrix(all_targets, all_preds)
    cm_normalized = confusion_matrix(all_targets, all_preds, normalize='true')
    logger.info(f"Confusion Matrix (Số lượng):\n{cm}")
    logger.info(f"Confusion Matrix (Phần trăm):\n{cm_normalized}")
    
    class_names = config.config.get('class_names', None)
    if class_names:
        cr = classification_report(all_targets, all_preds, target_names=class_names)
    else:
        cr = classification_report(all_targets, all_preds)
    logger.info(f"Classification Report:\n{cr}")

    # Vẽ và lưu Confusion Matrix thành file ảnh
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='.2%')
    plt.tight_layout()
    cm_save_path = str(config.log_dir / 'confusion_matrix.png')
    plt.savefig(cm_save_path)
    logger.info(f"Đã lưu ảnh Confusion Matrix tại: {cm_save_path}")


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