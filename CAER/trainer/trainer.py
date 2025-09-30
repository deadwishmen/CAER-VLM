import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import time
from torchvision import transforms

class Trainer(BaseTrainer):
    """
    Trainer class for models using ViltProcessor inputs
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start_epoch = time.time()
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            if inputs is None:  # Bỏ qua batch rỗng
                continue

            # Chuyển các tensor từ ViltProcessor sang device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)

            pixel_values_face = inputs['pixel_values_face'].to(self.device)
            pixel_values_context = inputs['pixel_values_context'].to(self.device)


            labels = labels.to(self.device)

            # Lấy ảnh PIL (list) để trực quan hóa
            face_imgs = inputs.get('face')  # List of PIL images
            body_imgs = inputs.get('body')
            context_imgs = inputs.get('context')

            self.optimizer.zero_grad()
            # Gọi mô hình với các tensor từ ViltProcessor
            output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            pixel_values_context=pixel_values_context,
                            pixel_values_face=pixel_values_face
            )
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, labels))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # Ghi ảnh vào TensorBoard nếu có
                # if face_imgs:
                #     face_tensors = [transforms.ToTensor()(img) for img in face_imgs]
                #     self.writer.add_image('face', make_grid(face_tensors, nrow=4, normalize=True))
                # if body_imgs:
                #     body_tensors = [transforms.ToTensor()(img) for img in body_imgs]
                #     self.writer.add_image('body', make_grid(body_tensors, nrow=4, normalize=True))
                # if context_imgs:
                #     context_tensors = [transforms.ToTensor()(img) for img in context_imgs]
                #     self.writer.add_image('context', make_grid(context_tensors, nrow=2, normalize=True))

                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.writer.add_histogram('grad_' + name, p.grad, bins='auto')

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        time_elapsed = time.time() - start_epoch
        print('Epoch completes in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.valid_data_loader):
                if inputs is None:  # Bỏ qua batch rỗng
                    continue
                labels = labels.to(self.device)
                # Chuyển các tensor từ ViltProcessor sang device
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                token_type_ids = inputs['token_type_ids'].to(self.device)

                pixel_values_face = inputs['pixel_values_face'].to(self.device)
                pixel_values_context = inputs['pixel_values_context'].to(self.device)

                # Lấy ảnh PIL (list) để trực quan hóa
                face_imgs = inputs.get('face')
                body_imgs = inputs.get('body')
                context_imgs = inputs.get('context')

                # Gọi mô hình
                output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            pixel_values_context=pixel_values_context,
                            pixel_values_face=pixel_values_face
                )
                loss = self.criterion(output, labels)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, labels))

                # Ghi ảnh vào TensorBoard nếu có
                # if face_imgs:
                #     face_tensors = [transforms.ToTensor()(img) for img in face_imgs]
                #     self.writer.add_image('face', make_grid(face_tensors, nrow=4, normalize=True))
                # if body_imgs:
                #     body_tensors = [transforms.ToTensor()(img) for img in body_imgs]
                #     self.writer.add_image('body', make_grid(body_tensors, nrow=4, normalize=True))
                # if context_imgs:
                #     context_tensors = [transforms.ToTensor()(img) for img in context_imgs]
                #     self.writer.add_image('context', make_grid(context_tensors, nrow=2, normalize=True))

        # Ghi histogram của tham số mô hình
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)