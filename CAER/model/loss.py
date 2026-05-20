import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(output, target):
    return F.cross_entropy(output, target, label_smoothing=0.1)


def prototype_loss(features, labels, prototypes, temperature=0.07):
    """
    Contrastive loss dựa trên distance đến prototype:
    - Kéo feature gần prototype cùng class (intra-class compact)
    - Đẩy feature xa prototype khác class (inter-class separation)
    
    Args:
        features:   [B, H] — image pooled features
        labels:     [B]    — ground truth class indices
        prototypes: [C, H] — prototype bank (C = num_classes)
        temperature: scalar — scaling factor
    """
    # Normalize để tính cosine similarity
    features_norm   = F.normalize(features, dim=1)    # [B, H]
    prototypes_norm = F.normalize(prototypes, dim=1)  # [C, H]

    # Similarity matrix: [B, C]
    sim_matrix = torch.matmul(features_norm, prototypes_norm.T) / temperature

    # Cross entropy trên similarity (giống InfoNCE)
    proto_loss = F.cross_entropy(sim_matrix, labels)
    return proto_loss


def combined_loss(output_dict, labels, prototypes_face, prototypes_context,
                  lambda_face=0.3, lambda_context=0.3, temperature=0.07):
    """
    Tổng hợp CE loss + Prototype loss từ 2 nhánh.
    Total = CE + λ_face * ProtoLoss_face + λ_context * ProtoLoss_context
    """
    # CE Loss trên classifier head
    ce = cross_entropy(output_dict['cat_pred'], labels)

    # Prototype Loss nhánh face
    proto_face = prototype_loss(
        output_dict['image_pool_face'],
        labels, prototypes_face, temperature
    )

    # Prototype Loss nhánh context
    proto_ctx = prototype_loss(
        output_dict['image_pool_context'],
        labels, prototypes_context, temperature
    )

    total = ce + lambda_face * proto_face + lambda_context * proto_ctx
    return total, {'ce': ce.item(), 'proto_face': proto_face.item(), 'proto_ctx': proto_ctx.item()}
