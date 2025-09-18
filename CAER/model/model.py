import torch
import torch.nn as nn
import os
from collections import OrderedDict
from base import BaseModel
from transformers import ViltProcessor, ViltModel, ViltConfig

class ViLTModule(nn.Module):
    """Module trích xuất đặc trưng sử dụng ViLT."""
    def __init__(self, vilt_model_name="dandelin/vilt-b32-mlm", num_classes=7, num_attention_heads=8, num_layer_decoder=4, dim_feedforward=2048, new_text_max_len_for_model=256, cache_dir=None, freeze_vilt_base=True, dropout_rate=0.1):
        super().__init__()

        self.vilt_model_name = vilt_model_name
        offline_mode = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
        self.vilt = ViltModel.from_pretrained(vilt_model_name, local_files_only=offline_mode, cache_dir=cache_dir)
        
        # Resize embedding
        text_embed_module = self.vilt.embeddings.text_embeddings
        old_text_pos_embed_layer = text_embed_module.position_embeddings
        self.old_num_text_embeddings = old_text_pos_embed_layer.num_embeddings
        if new_text_max_len_for_model > self.old_num_text_embeddings:
            print(f"Resizing VILT text position embeddings layer from {self.old_num_text_embeddings} to {new_text_max_len_for_model}")
            new_pos_embed_layer = nn.Embedding(new_text_max_len_for_model, self.vilt.config.hidden_size)
            new_pos_embed_layer.weight.data.normal_(mean=0.0, std=self.vilt.config.initializer_range)
            len_to_copy = min(self.old_num_text_embeddings, new_text_max_len_for_model)
            new_pos_embed_layer.weight.data[:len_to_copy, :] = old_text_pos_embed_layer.weight.data[:len_to_copy, :]
            text_embed_module.position_embeddings = new_pos_embed_layer
            if hasattr(self.vilt.config, 'text_config'):
                self.vilt.config.text_config.max_position_embeddings = new_text_max_len_for_model
            else:
                self.vilt.config.max_position_embeddings = new_text_max_len_for_model
            current_device = text_embed_module.position_ids.device
            text_embed_module.position_ids = nn.Parameter(
                torch.arange(new_text_max_len_for_model, device=current_device).expand((1, -1)),
                requires_grad=False
            )

        if freeze_vilt_base:
            for name, param in self.vilt.named_parameters():
                is_new_text_pos_embed_weight = "embeddings.text_embeddings.position_embeddings.weight" in name
                if is_new_text_pos_embed_weight and new_text_max_len_for_model > self.old_num_text_embeddings:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.vilt.config.hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.fusion_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layer_decoder)

        # Classifier
        classifier_input_dim = 4 * self.vilt.config.hidden_size
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, self.vilt.config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.vilt.config.hidden_size, num_classes)
        )
        print(f"ViltEmoticModel with TransformerDecoderLayer initialized. Classifier input dim: {classifier_input_dim}")

    def _masked_mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Hàm helper để tính mean pooling có bỏ qua padding."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_hidden_state = last_hidden_state * mask
        summed_state = torch.sum(masked_hidden_state, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed_state / summed_mask
    
    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values_context, pixel_values_face):
        # Trích xuất đặc trưng cho context
        vilt_outputs_context = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values_context,
            return_dict=True
        )
        last_hidden_state_context = vilt_outputs_context.last_hidden_state
        cls_features_context = last_hidden_state_context[:, 0, :]

        # Trích xuất đặc trưng cho face (không sử dụng body)
        vilt_outputs_face = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values_face,
            return_dict=True
        )
        last_hidden_state_face = vilt_outputs_face.last_hidden_state
        cls_features_face = last_hidden_state_face[:, 0, :]

        # Fusion với Transformer Decoder
        fused_sequences = self.fusion_decoder(
            tgt=last_hidden_state_face,
            memory=last_hidden_state_context
        )
        fused_cls_output = fused_sequences[:, 0, :]

        # Pooling trên đặc trưng văn bản
        num_patches = (self.vilt.config.image_size // self.vilt.config.patch_size) ** 2
        text_embedding_start_index = 1 + num_patches
        text_hidden_state_context = last_hidden_state_context[:, text_embedding_start_index:, :]
        text_hidden_state_face = last_hidden_state_face[:, text_embedding_start_index:, :]
        pooled_features_context = self._masked_mean_pooling(text_hidden_state_context, attention_mask)
        pooled_features_face = self._masked_mean_pooling(text_hidden_state_face, attention_mask)

        # Kết hợp đặc trưng
        combined_features = torch.cat(
            (
                cls_features_face,        # Đặc trưng [CLS] gốc của face
                fused_cls_output,         # Đặc trưng [CLS] sau fusion
                pooled_features_face,     # Đặc trưng pooling của văn bản face
                pooled_features_context   # Đặc trưng pooling của văn bản context
            ), 
            dim=1
        )

        # Dự đoán
        cat_pred = self.classifier_head(combined_features)
        return cat_pred

class CAERSVLMNet(BaseModel):
    """Mô hình tổng thể, sử dụng ViLTModule."""
    def __init__(self, num_classes=7,
                 vilt_model_name="dandelin/vilt-b32-mlm",
                 dropout_rate=0.1,
                 text_max_len=256,
                 freeze_vilt_base=True):
        super().__init__()
        
        self.ViLT_model = ViLTModule(
            vilt_model_name=vilt_model_name,
            num_classes=num_classes,
            new_text_max_len_for_model=text_max_len,
            freeze_vilt_base=freeze_vilt_base,
            dropout_rate=dropout_rate
        )

    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values_context, pixel_values_face):
        return self.ViLT_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values_context=pixel_values_context,
            pixel_values_face=pixel_values_face
        )