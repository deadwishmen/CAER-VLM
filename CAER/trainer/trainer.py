import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.loss import combined_loss
import time


class Trainer(BaseTrainer):
    """
    Trainer class cho ViLT-based emotion recognition model.

    Cải tiến so với bản gốc:
    ─────────────────────────────────────────────────────────────
    1. Tích hợp Center Loss qua centers_face / centers_context
    2. Tích hợp SupCon Loss (bật/tắt qua config)
    3. Log đầy đủ tất cả loss components lên TensorBoard
    4. Gradient clipping giữ nguyên
    ─────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader

        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # Loss component names để track
        loss_keys = ["loss", "ce", "proto_face", "proto_ctx", "supcon", "center"]
        self.train_metrics = MetricTracker(
            *loss_keys, *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )

    # ── Lấy config trainer với fallback ──────────────────────────
    def _tcfg(self, key, default):
        return self.config["trainer"].get(key, default)

    # ── Train một epoch ──────────────────────────────────────────
    def _train_epoch(self, epoch):
        start_epoch = time.time()
        self.model.train()
        self.train_metrics.reset()

        is_parallel  = isinstance(self.model, torch.nn.DataParallel)
        model_module = self.model.module if is_parallel else self.model

        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            if inputs is None:
                continue

            # ── Chuyển sang device ────────────────────────────────
            input_ids        = inputs["input_ids"].to(self.device)
            attention_mask   = inputs["attention_mask"].to(self.device)
            token_type_ids   = inputs["token_type_ids"].to(self.device)
            pixel_values_face    = inputs["pixel_values_face"].to(self.device)
            pixel_values_context = inputs["pixel_values_context"].to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # ── Forward ───────────────────────────────────────────
            output_dict = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values_context=pixel_values_context,
                pixel_values_face=pixel_values_face,
                labels=labels,
            )

            # ── Combined Loss ─────────────────────────────────────
            loss, components = combined_loss(
                output_dict,
                labels,
                prototypes_face    = model_module.ViLT_model.prototypes_face,
                prototypes_context = model_module.ViLT_model.prototypes_context,
                centers_face       = model_module.ViLT_model.centers_face,
                centers_context    = model_module.ViLT_model.centers_context,
                lambda_face        = self._tcfg("lambda_face",    0.3),
                lambda_context     = self._tcfg("lambda_context",  0.3),
                lambda_supcon      = self._tcfg("lambda_supcon",   0.5),
                lambda_center      = self._tcfg("lambda_center",   0.01),
                temperature        = self._tcfg("proto_temp",      0.07),
                use_supcon         = self._tcfg("use_supcon",      True),
                use_center         = self._tcfg("use_center",      True),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # ── Logging ───────────────────────────────────────────
            step = (epoch - 1) * self.len_epoch + batch_idx
            self.writer.set_step(step)
            for k, v in components.items():
                self.writer.add_scalar(f"loss/{k}", v)
            self.train_metrics.update("loss",       components["total"])
            self.train_metrics.update("ce",         components["ce"])
            self.train_metrics.update("proto_face", components["proto_face"])
            self.train_metrics.update("proto_ctx",  components["proto_ctx"])
            self.train_metrics.update("supcon",     components["supcon"])
            self.train_metrics.update("center",     components["center"])

            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(output_dict["cat_pred"], labels)
                )

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.4f} | CE: {:.4f} | "
                    "SupCon: {:.4f} | Center: {:.4f}".format(
                        epoch,
                        self._progress(batch_idx),
                        components["total"],
                        components["ce"],
                        components["supcon"],
                        components["center"],
                    )
                )
                # Gradient histogram
                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.writer.add_histogram("grad_" + name, p.grad, bins="auto")

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        elapsed = time.time() - start_epoch
        print(
            "Epoch {} done in {:.0f}m {:.0f}s | "
            "val_acc={:.4f}".format(
                epoch,
                elapsed // 60,
                elapsed % 60,
                log.get("val_accuracy", 0),
            )
        )
        return log

    # ── Validation ───────────────────────────────────────────────
    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.valid_data_loader):
                if inputs is None:
                    continue

                input_ids        = inputs["input_ids"].to(self.device)
                attention_mask   = inputs["attention_mask"].to(self.device)
                token_type_ids   = inputs["token_type_ids"].to(self.device)
                pixel_values_face    = inputs["pixel_values_face"].to(self.device)
                pixel_values_context = inputs["pixel_values_context"].to(self.device)
                labels = labels.to(self.device)

                output_dict = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    pixel_values_context=pixel_values_context,
                    pixel_values_face=pixel_values_face,
                )
                output = output_dict["cat_pred"]
                loss = self.criterion(output, labels)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, labels))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)