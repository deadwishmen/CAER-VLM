import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(output, target):
    return F.cross_entropy(output, target, label_smoothing=0.1)


def prototype_loss(features, labels, prototypes, temperature=0.07, margin=0.2):
    """
    Contrastive loss với additive angular margin (ArcFace-style):
    - Trừ margin m khỏi cosine similarity của đúng class → buộc model
      phải học tighter cluster (intra-class compact hơn)
    - Đẩy feature xa prototype khác class (inter-class separation)

    Công thức:
        sim[i, y_i] = cos(θ) - m      (đúng class bị phạt)
        sim[i, j≠y_i] = cos(θ)        (class sai giữ nguyên)
        loss = CrossEntropy(sim / τ)

    Args:
        features:    [B, H] — image pooled features
        labels:      [B]    — ground truth class indices
        prototypes:  [C, H] — prototype bank (C = num_classes)
        temperature: scalar — scaling factor τ
        margin:      scalar — additive margin m (thường 0.1–0.5)
    """
    # Normalize để tính cosine similarity
    features_norm   = F.normalize(features, dim=1)    # [B, H]
    prototypes_norm = F.normalize(prototypes, dim=1)  # [C, H]

    # Similarity matrix: [B, C]
    sim_matrix = torch.matmul(features_norm, prototypes_norm.T)  # [B, C]

    # --- Áp dụng margin vào đúng class ---
    # Tạo one-hot mask tại vị trí ground-truth
    one_hot = torch.zeros_like(sim_matrix)                        # [B, C]
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)                 # 1 tại y_i

    # Trừ margin m chỉ ở cột đúng class
    sim_matrix = sim_matrix - margin * one_hot                    # [B, C]

    # Scale bằng temperature rồi tính CE loss
    proto_loss = F.cross_entropy(sim_matrix / temperature, labels)
    return proto_loss


def combined_loss(output_dict, labels, prototypes_face, prototypes_context,
                  lambda_face=0.3, lambda_context=0.3,
                  temperature=0.07, margin=0.2):
    """
    Tổng hợp CE loss + Prototype loss (có margin) từ 2 nhánh.
    Total = CE + λ_face * ProtoLoss_face + λ_context * ProtoLoss_context
    """
    # CE Loss trên classifier head
    ce = cross_entropy(output_dict['cat_pred'], labels)

    # Prototype Loss nhánh face
    proto_face = prototype_loss(
        output_dict['image_pool_face'],
        labels, prototypes_face, temperature, margin
    )

    # Prototype Loss nhánh context
    proto_ctx = prototype_loss(
        output_dict['image_pool_context'],
        labels, prototypes_context, temperature, margin
    )

    total = ce + lambda_face * proto_face + lambda_context * proto_ctx
    return total, {
        'ce':         ce.item(),
        'proto_face': proto_face.item(),
        'proto_ctx':  proto_ctx.item(),
    }