import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from base import BaseModel
from transformers import ViltModel


class ViLTModule(nn.Module):
    """
    Module trích xuất đặc trưng sử dụng ViLT.

    Cải tiến so với bản gốc:
    ─────────────────────────────────────────────────────────────
    1. Projection Head (MLP nhỏ) trước prototype/contrastive loss
       → tách không gian classifier ra khỏi không gian metric
    2. Learnable Center Bank (cho Center Loss)
       → kéo intra-class feature lại gần nhau
    3. Prototype khởi tạo bằng running-mean thay vì zeros
       → prototype hội tụ nhanh hơn ở epoch đầu
    4. Prototype momentum tự tăng dần (warm-up)
       → giai đoạn đầu học nhanh, sau đó ổn định
    ─────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        vilt_model_name="dandelin/vilt-b32-mlm",
        num_classes=7,
        num_attention_heads=8,
        num_layer_decoder=2,
        dim_feedforward=1024,
        new_text_max_len_for_model=256,
        cache_dir=None,
        freeze_vilt_base=True,
        dropout_rate=0.1,
        use_context_image=True,
        use_face_image=True,
        use_fusion=True,
        prototype_momentum=0.99,
        prototype_temp=0.07,
        proj_dim=256,           # chiều của projection head
    ):
        super().__init__()

        self.vilt_model_name = vilt_model_name
        offline_mode = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
        self.vilt = ViltModel.from_pretrained(
            vilt_model_name, local_files_only=offline_mode, cache_dir=cache_dir
        )

        # ── Resize text position embeddings ──────────────────────
        text_embed_module = self.vilt.embeddings.text_embeddings
        old_text_pos_embed_layer = text_embed_module.position_embeddings
        self.old_num_text_embeddings = old_text_pos_embed_layer.num_embeddings
        if new_text_max_len_for_model > self.old_num_text_embeddings:
            print(
                f"Resizing VILT text position embeddings: "
                f"{self.old_num_text_embeddings} → {new_text_max_len_for_model}"
            )
            new_pos_embed_layer = nn.Embedding(
                new_text_max_len_for_model, self.vilt.config.hidden_size
            )
            new_pos_embed_layer.weight.data.normal_(
                mean=0.0, std=self.vilt.config.initializer_range
            )
            len_to_copy = min(self.old_num_text_embeddings, new_text_max_len_for_model)
            new_pos_embed_layer.weight.data[:len_to_copy, :] = (
                old_text_pos_embed_layer.weight.data[:len_to_copy, :]
            )
            text_embed_module.position_embeddings = new_pos_embed_layer
            if hasattr(self.vilt.config, "text_config"):
                self.vilt.config.text_config.max_position_embeddings = new_text_max_len_for_model
            else:
                self.vilt.config.max_position_embeddings = new_text_max_len_for_model
            current_device = text_embed_module.position_ids.device
            text_embed_module.position_ids = nn.Parameter(
                torch.arange(new_text_max_len_for_model, device=current_device).expand((1, -1)),
                requires_grad=False,
            )

        if freeze_vilt_base:
            for name, param in self.vilt.named_parameters():
                is_new_pos = "embeddings.text_embeddings.position_embeddings.weight" in name
                if is_new_pos and new_text_max_len_for_model > self.old_num_text_embeddings:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        hidden_size = self.vilt.config.hidden_size   # 768

        # ── Transformer Decoder (fusion) ─────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fusion_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layer_decoder)

        # ── Classifier Head ──────────────────────────────────────
        classifier_input_dim = 4 * hidden_size
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

        # ── [NEW] Projection Head ────────────────────────────────
        # Dùng để tính prototype loss / supcon loss
        # Tách không gian classification ra khỏi metric space
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim),
        )
        self.proj_dim = proj_dim

        # ── [NEW] Learnable Center Bank ──────────────────────────
        # register_buffer → lưu vào checkpoint, không tính grad cho optimizer chính
        # Nhưng update thủ công bằng gradient (center loss)
        self.register_buffer("centers_face",    torch.zeros(num_classes, hidden_size))
        self.register_buffer("centers_context", torch.zeros(num_classes, hidden_size))

        # ── Prototype Bank (EMA) ─────────────────────────────────
        # Dùng proj_dim để match projection head output
        self.register_buffer("prototypes_face",    torch.zeros(num_classes, proj_dim))
        self.register_buffer("prototypes_context", torch.zeros(num_classes, proj_dim))
        self.register_buffer("prototype_counts",   torch.zeros(num_classes))

        # Prototype momentum khởi đầu thấp để warmup nhanh
        self.prototype_momentum = prototype_momentum
        self._proto_step = 0    # dùng để warm-up momentum

        self.use_context_image = use_context_image
        self.use_face_image = use_face_image
        self.use_fusion = use_fusion

        if self.use_fusion:
            assert self.use_context_image and self.use_face_image, \
                "use_fusion yêu cầu cả context và face."

        print(
            f"ViLTModule initialized | classifier_input_dim={classifier_input_dim} "
            f"| proj_dim={proj_dim}"
        )

    # ── Helpers ───────────────────────────────────────────────────
    def _masked_mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        count  = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def _get_image_pooled(self, last_hidden_state, text_len):
        return last_hidden_state[:, text_len:, :].mean(dim=1)

    def _get_text_pooled(self, last_hidden_state, attention_mask, text_len):
        text_hidden = last_hidden_state[:, :text_len, :]
        text_mask   = attention_mask[:, :text_len]
        return self._masked_mean_pooling(text_hidden, text_mask)

    @torch.no_grad()
    def _update_prototypes(self, proj_features, labels, prototype_bank):
        """
        EMA update trên projection features (đã normalize).
        Momentum warm-up: bắt đầu từ 0.9, tăng dần đến target.
        """
        self._proto_step += 1
        # Warm-up 500 step đầu: momentum tăng từ 0.5 → target
        warmup_steps = 500
        if self._proto_step < warmup_steps:
            m = 0.5 + (self.prototype_momentum - 0.5) * (self._proto_step / warmup_steps)
        else:
            m = self.prototype_momentum

        feats_norm = F.normalize(proj_features, dim=1)   # [B, proj_dim]

        for c in range(prototype_bank.shape[0]):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            class_mean = feats_norm[mask].mean(dim=0)   # [proj_dim]

            if prototype_bank[c].norm() < 1e-6:
                # Lần đầu: khởi tạo thẳng, không EMA
                prototype_bank[c] = class_mean
            else:
                prototype_bank[c] = m * prototype_bank[c] + (1 - m) * class_mean

    @torch.no_grad()
    def _update_centers(self, features, labels, center_bank, lr_center=0.5):
        """
        Update center bank bằng gradient đơn giản (không qua optimizer).
        delta_c = mean(feat - center) over class c
        """
        for c in range(center_bank.shape[0]):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            diff = features[mask].mean(dim=0) - center_bank[c]
            center_bank[c] = center_bank[c] + lr_center * diff

    # ── Forward ───────────────────────────────────────────────────
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values_context,
        pixel_values_face,
        labels=None,
    ):
        batch_size, text_len = input_ids.shape
        hidden_size = self.vilt.config.hidden_size
        device = input_ids.device

        last_hidden_state_context = None
        last_hidden_state_face = None

        need_context = self.use_context_image or self.use_fusion
        need_face    = self.use_face_image    or self.use_fusion

        if need_context and need_face:
            # Gom batch → 1 lần forward qua ViLT
            combined_input_ids       = torch.cat([input_ids,          input_ids],          dim=0)
            combined_attention_mask  = torch.cat([attention_mask,     attention_mask],     dim=0)
            combined_token_type_ids  = torch.cat([token_type_ids,     token_type_ids],     dim=0)
            combined_pixel_values    = torch.cat([pixel_values_context, pixel_values_face], dim=0)

            combined_outputs = self.vilt(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                token_type_ids=combined_token_type_ids,
                pixel_values=combined_pixel_values,
                return_dict=True,
            )
            last_hidden_state_context, last_hidden_state_face = torch.split(
                combined_outputs.last_hidden_state, batch_size, dim=0
            )
        elif need_context:
            out = self.vilt(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, pixel_values=pixel_values_context,
                            return_dict=True)
            last_hidden_state_context = out.last_hidden_state
        elif need_face:
            out = self.vilt(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, pixel_values=pixel_values_face,
                            return_dict=True)
            last_hidden_state_face = out.last_hidden_state

        # ── Fusion ────────────────────────────────────────────────
        if self.use_fusion:
            face_patches    = last_hidden_state_face[:, text_len:, :]
            context_patches = last_hidden_state_context[:, text_len:, :]
            fused_sequences = self.fusion_decoder(tgt=face_patches, memory=context_patches)
            fused_image_output = fused_sequences.mean(dim=1)
        else:
            fused_image_output = torch.zeros(batch_size, hidden_size, device=device)

        # ── Image Pool Features ───────────────────────────────────
        image_pool_face = (
            self._get_image_pooled(last_hidden_state_face, text_len)
            if self.use_face_image
            else torch.zeros(batch_size, hidden_size, device=device)
        )
        image_pool_context = (
            self._get_image_pooled(last_hidden_state_context, text_len)
            if self.use_context_image
            else torch.zeros(batch_size, hidden_size, device=device)
        )

        # ── Text Pool ─────────────────────────────────────────────
        if self.use_context_image and last_hidden_state_context is not None:
            pooled_text = self._get_text_pooled(last_hidden_state_context, attention_mask, text_len)
        elif self.use_face_image and last_hidden_state_face is not None:
            pooled_text = self._get_text_pooled(last_hidden_state_face, attention_mask, text_len)
        else:
            pooled_text = torch.zeros(batch_size, hidden_size, device=device)

        # ── [NEW] Projection (dùng cho prototype/supcon loss) ─────
        proj_face    = self.proj_head(image_pool_face)      # [B, proj_dim]
        proj_context = self.proj_head(image_pool_context)   # [B, proj_dim]

        # ── Update Prototype & Center (training only) ─────────────
        if self.training and labels is not None:
            self._update_prototypes(proj_face,    labels, self.prototypes_face)
            self._update_prototypes(proj_context, labels, self.prototypes_context)
            self._update_centers(image_pool_face,    labels, self.centers_face)
            self._update_centers(image_pool_context, labels, self.centers_context)

        # ── Classifier ────────────────────────────────────────────
        combined_features = torch.cat(
            [image_pool_face, image_pool_context, fused_image_output, pooled_text],
            dim=1,
        )   # [B, 4*H]
        cat_pred = self.classifier_head(combined_features)

        return {
            "cat_pred":           cat_pred,           # [B, num_classes]
            "image_pool_face":    proj_face,           # [B, proj_dim] — dùng cho loss
            "image_pool_context": proj_context,        # [B, proj_dim]
            "feat_face_raw":      image_pool_face,     # [B, H] — dùng cho center loss
            "feat_ctx_raw":       image_pool_context,  # [B, H]
        }


class CAERSVLMNet(BaseModel):
    """Mô hình tổng thể, sử dụng ViLTModule."""

    def __init__(
        self,
        num_classes=7,
        vilt_model_name="dandelin/vilt-b32-mlm",
        dropout_rate=0.1,
        text_max_len=256,
        freeze_vilt_base=True,
        use_context_image=True,
        use_face_image=True,
        use_fusion=True,
        prototype_momentum=0.99,
        proj_dim=256,
    ):
        super().__init__()

        self.ViLT_model = ViLTModule(
            vilt_model_name=vilt_model_name,
            num_classes=num_classes,
            new_text_max_len_for_model=text_max_len,
            freeze_vilt_base=freeze_vilt_base,
            dropout_rate=dropout_rate,
            use_context_image=use_context_image,
            use_face_image=use_face_image,
            use_fusion=use_fusion,
            prototype_momentum=prototype_momentum,
            proj_dim=proj_dim,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values_context,
        pixel_values_face,
        labels=None,
    ):
        return self.ViLT_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values_context=pixel_values_context,
            pixel_values_face=pixel_values_face,
            labels=labels,
        )