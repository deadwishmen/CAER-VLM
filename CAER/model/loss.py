import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Cross-Entropy với label smoothing
# ─────────────────────────────────────────────
def cross_entropy(output, target, label_smoothing=0.1):
    return F.cross_entropy(output, target, label_smoothing=label_smoothing)


# ─────────────────────────────────────────────
# 2. Center Loss
# Kéo mỗi feature về prototype trung tâm của class mình
# → giảm intra-class variance trực tiếp
# ─────────────────────────────────────────────
def center_loss(features, labels, centers):
    """
    Args:
        features: [B, H] — feature vectors (KHÔNG cần normalize)
        labels:   [B]    — class indices
        centers:  [C, H] — class center bank (learnable hoặc EMA)
    Returns:
        scalar loss
    """
    # Lấy center của từng sample theo label
    centers_batch = centers[labels]          # [B, H]
    diff = features - centers_batch          # [B, H]
    loss = (diff ** 2).sum(dim=1).mean()     # scalar
    return loss * 0.5


# ─────────────────────────────────────────────
# 3. Supervised Contrastive Loss (SupCon)
# Đẩy feature của cùng class lại gần nhau,
# đẩy feature khác class ra xa — giải quyết
# bài toán "cùng nhãn nhưng feature khác nhau"
# và "khác nhãn nhưng feature giống nhau"
# ─────────────────────────────────────────────
def supcon_loss(features, labels, temperature=0.07):
    """
    Simplified Supervised Contrastive Loss.
    Args:
        features: [B, H] — L2-normalized feature vectors
        labels:   [B]    — class indices
        temperature: scalar
    Returns:
        scalar loss
    """
    device = features.device
    B = features.shape[0]

    # Normalize
    features = F.normalize(features, dim=1)   # [B, H]

    # Similarity matrix
    sim = torch.matmul(features, features.T) / temperature   # [B, B]

    # Mask: positive pairs = cùng class, loại diagonal
    labels_col = labels.unsqueeze(1)          # [B, 1]
    labels_row = labels.unsqueeze(0)          # [1, B]
    pos_mask = (labels_col == labels_row).float()   # [B, B]
    pos_mask.fill_diagonal_(0)                # bỏ self

    # Nếu một sample không có positive pair trong batch → skip
    has_pos = pos_mask.sum(dim=1) > 0
    if not has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Loại diagonal khỏi denominator
    diag_mask = 1 - torch.eye(B, device=device)

    # Log-sum-exp trick: log Σ exp(sim_neg)
    exp_sim = torch.exp(sim) * diag_mask      # [B, B]
    log_denom = torch.log(exp_sim.sum(dim=1) + 1e-9)   # [B]

    # Loss = -mean over positives
    log_prob = sim - log_denom.unsqueeze(1)   # [B, B]
    loss = -(pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-9)
    loss = loss[has_pos].mean()
    return loss


# ─────────────────────────────────────────────
# 4. Prototype Loss (giữ lại, cải tiến)
# Dùng cosine similarity + cross-entropy
# nhưng thêm guard khi prototype chưa warmup
# ─────────────────────────────────────────────
def prototype_loss(features, labels, prototypes, temperature=0.07):
    """
    Args:
        features:   [B, H]
        labels:     [B]
        prototypes: [C, H]
        temperature: scalar
    Returns:
        scalar loss, hoặc 0 nếu prototype chưa warmup
    """
    # Nếu prototype vẫn còn zero (chưa được update) → skip
    proto_norm = prototypes.norm(dim=1)       # [C]
    if proto_norm.sum() < 1e-6:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    features_norm   = F.normalize(features,   dim=1)   # [B, H]
    prototypes_norm = F.normalize(prototypes, dim=1)   # [C, H]

    sim_matrix = torch.matmul(features_norm, prototypes_norm.T) / temperature   # [B, C]
    return F.cross_entropy(sim_matrix, labels)


# ─────────────────────────────────────────────
# 5. Combined Loss (tổng hợp tất cả)
# ─────────────────────────────────────────────
def combined_loss(
    output_dict,
    labels,
    prototypes_face,
    prototypes_context,
    centers_face=None,
    centers_context=None,
    lambda_face=0.3,
    lambda_context=0.3,
    lambda_supcon=0.5,
    lambda_center=0.01,
    temperature=0.07,
    use_supcon=True,
    use_center=True,
):
    """
    Total = CE
          + λ_face    * ProtoLoss(face)
          + λ_context * ProtoLoss(context)
          + λ_supcon  * SupConLoss(face + context concat)
          + λ_center  * CenterLoss(face)      [nếu có centers]
          + λ_center  * CenterLoss(context)   [nếu có centers]

    Trả về:
        total (scalar, có grad)
        components (dict, detached float — để log)
    """
    device = labels.device

    # ── CE ──────────────────────────────────
    ce = cross_entropy(output_dict['cat_pred'], labels)

    # ── Prototype ───────────────────────────
    proto_face = prototype_loss(
        output_dict['image_pool_face'], labels, prototypes_face, temperature
    )
    proto_ctx = prototype_loss(
        output_dict['image_pool_context'], labels, prototypes_context, temperature
    )

    # ── SupCon ──────────────────────────────
    supcon = torch.tensor(0.0, device=device)
    if use_supcon:
        # Concat face & context để tạo "view" pair, giống SupCon hai view
        feat_face = F.normalize(output_dict['image_pool_face'],    dim=1)
        feat_ctx  = F.normalize(output_dict['image_pool_context'], dim=1)

        # Stack thành [2B, H], labels lặp đôi
        all_feats  = torch.cat([feat_face, feat_ctx],  dim=0)   # [2B, H]
        all_labels = torch.cat([labels,    labels],    dim=0)   # [2B]
        supcon = supcon_loss(all_feats, all_labels, temperature)

    # ── Center Loss ─────────────────────────
    closs = torch.tensor(0.0, device=device)
    if use_center and centers_face is not None and centers_context is not None:
        closs = (
            center_loss(output_dict['feat_face_raw'],    labels, centers_face)
            + center_loss(output_dict['feat_ctx_raw'], labels, centers_context)
        ) * 0.5

    # ── Total ───────────────────────────────
    total = (
        ce
        + lambda_face    * proto_face
        + lambda_context * proto_ctx
        + lambda_supcon  * supcon
        + lambda_center  * closs
    )

    components = {
        'ce':         ce.item(),
        'proto_face': proto_face.item(),
        'proto_ctx':  proto_ctx.item(),
        'supcon':     supcon.item(),
        'center':     closs.item(),
        'total':      total.item(),
    }
    return total, components