import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
from base import BaseModel

# ==============================================================================
# 1. CÁC MODULE ĐƠN LẺ VÀ HÀM HỖ TRỢ
# ==============================================================================

class FaceEmotionCNN(nn.Module):
    """
    Mạng CNN để nhận dạng cảm xúc khuôn mặt.
    CẬP NHẬT: Kiến trúc nn.Sequential được sắp xếp lại để khớp 100%
    với thứ tự các lớp trong code gốc (Conv->Dropout->BN->Pool->ReLU).
    """
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Conv -> BN -> Pool -> ReLU
            nn.Conv2d(1, 8, kernel_size=3), nn.BatchNorm2d(8), nn.MaxPool2d(2, 1), nn.ReLU(inplace=True),
            # Block 2: Conv -> Dropout -> BN -> Pool -> ReLU
            nn.Conv2d(8, 16, kernel_size=3), nn.Dropout(0.3), nn.BatchNorm2d(16), nn.MaxPool2d(2, 1), nn.ReLU(inplace=True),
            # Block 3: Conv -> BN -> Pool -> ReLU
            nn.Conv2d(16, 32, kernel_size=3), nn.BatchNorm2d(32), nn.MaxPool2d(2, 1), nn.ReLU(inplace=True),
            # Block 4: Conv -> Dropout -> BN -> Pool -> ReLU
            nn.Conv2d(32, 64, kernel_size=3), nn.Dropout(0.3), nn.BatchNorm2d(64), nn.MaxPool2d(2, 1), nn.ReLU(inplace=True),
            # Block 5: Conv -> BN -> Pool -> ReLU
            nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
            # Block 6: Conv -> Dropout -> BN -> Pool -> ReLU
            nn.Conv2d(128, 256, kernel_size=3), nn.Dropout(0.3), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
            # Block 7: Conv -> Dropout -> BN -> Pool -> ReLU
            nn.Conv2d(256, 256, kernel_size=3), nn.Dropout(0.3), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Block FC 1: Linear -> Dropout -> ReLU
            nn.Linear(1024, 512), nn.Dropout(0.3), nn.ReLU(inplace=True),
            # Block FC 2: Linear -> Dropout -> ReLU
            nn.Linear(512, 256), nn.Dropout(0.3), nn.ReLU(inplace=True),
            # Block FC 3 (lớp cuối)
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Thay thế hàm cũ bằng hàm này trong file model/model.py

def create_backbone(name: str):
    """
    Hàm "nhà máy" (factory) hợp nhất để tạo các backbone khác nhau.
    Luôn trả về một tuple: (module_backbone_trả_về_vector_1D, số_chiều_đặc_trưng).
    """
    if name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        feature_dim = model.fc.in_features  # 512
        model.fc = nn.Identity()
        return model, feature_dim
    
    elif name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        feature_dim = model.fc.in_features # 2048
        model.fc = nn.Identity()
        return model, feature_dim
        
    elif name == 'resnet50_places':
        model = models.resnet50(num_classes=365)
        # Sử dụng URL ổn định hơn
        url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
        print(f"Đang tiến hành tải trọng số cho '{name}'...")

        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("✅ Đã tải thành công trọng số ResNet50-Places365.")


        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    elif name == 'swin_t':
        model = models.swin_t(weights='DEFAULT')
        feature_dim = model.head.in_features # 768
        model.head = nn.Identity()
        return model, feature_dim
        
    else:
        raise ValueError(f"Backbone '{name}' không được hỗ trợ. Vui lòng chọn: 'resnet18', 'resnet50', 'resnet50_places', 'swin_t'.")
# ==============================================================================
# 2. KIẾN TRÚC KẾT HỢP
# ==============================================================================

class FeatureExtractors(nn.Module):
    """
    Module chứa 3 bộ trích xuất đặc trưng cho Face, Body, và Context.
    CẬP NHẬT: Thêm lại đầy đủ logic tải và chuyển đổi key cho Face Model.
    """
    def __init__(self, face_model_path, num_classes=7, body_backbone='swin_t', context_backbone='resnet50_places',
                 freeze_backbones=True):
        """
        :param face_model_path: Đường dẫn đến file trọng số Face Model.
        :param num_classes: Số lớp đầu ra của Face Model.
        :param body_backbone: Tên của backbone cho Body.
        :param context_backbone: Tên của backbone cho Context.
        :param freeze_backbones: Nếu True, đóng băng các backbone để không cập nhật trọng số trong quá trình huấn luyện.
        """
        super().__init__()
        
        # --- Face Extractor (Tùy chỉnh) ---
        # 1. Khởi tạo kiến trúc FaceEmotionCNN mới
        full_face_model = FaceEmotionCNN(num_classes=num_classes)
        
        # 2. Bắt đầu logic tải và chuyển đổi key từ file checkpoint cũ
        print(f"Bắt đầu tải và chuyển đổi trọng số cho Face Model từ: {face_model_path}")
        try:
            old_state_dict = torch.load(face_model_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            
            # Ánh xạ từ tên lớp cũ sang tên lớp mới trong nn.Sequential
            key_map = {
                'cnn1': 'features.0',  'cnn1_bn': 'features.1',
                'cnn2': 'features.4',  'cnn2_bn': 'features.6',
                'cnn3': 'features.9',  'cnn3_bn': 'features.10',
                'cnn4': 'features.13', 'cnn4_bn': 'features.15',
                'cnn5': 'features.18', 'cnn5_bn': 'features.19',
                'cnn6': 'features.22', 'cnn6_bn': 'features.24',
                'cnn7': 'features.27', 'cnn7_bn': 'features.29',
                'fc1': 'classifier.1', 'fc2': 'classifier.4', 'fc3': 'classifier.7',
            }
            
            for old_key, value in old_state_dict.items():
                parts = old_key.split('.')
                layer_name = parts[0]
                if layer_name in key_map:
                    param_type = '.'.join(parts[1:])
                    new_layer_name = key_map[layer_name]
                    new_key = f"{new_layer_name}.{param_type}"
                    new_state_dict[new_key] = value
                else:
                    # Giữ lại các key không cần chuyển đổi nếu có
                    new_state_dict[old_key] = value
            
            # Tải state_dict đã được chuyển đổi vào model mới
            full_face_model.load_state_dict(new_state_dict)
            print("=> Đã tải và chuyển đổi thành công trọng số cho Face Model.")

        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy file trọng số cho Face Model tại '{face_model_path}'. Sử dụng trọng số ngẫu nhiên.")
        except Exception as e:
            print(f"Lỗi khi tải trọng số Face Model: {e}. Sử dụng trọng số ngẫu nhiên.")

        # 3. Tách phần trích xuất đặc trưng và phần đầu của classifier
        self.face_features = full_face_model.features
        self.face_classifier_head = full_face_model.classifier[:-1] # Bỏ lớp fc cuối cùng
        self.face_dim = 256 # Kích thước đặc trưng cuối cùng của face stream

        # --- Body and Context Extractors (Linh hoạt) ---
        print(f"Sử dụng backbone '{body_backbone}' cho Body.")
        self.body_extractor, self.body_dim = create_backbone(body_backbone)
        
        print(f"Sử dụng backbone '{context_backbone}' cho Context.")
        self.context_extractor, self.context_dim = create_backbone(context_backbone)


        if freeze_backbones:
            print("--- ĐÓNG BĂNG CÁC BACKBONE ---")
            # Đóng băng phần trích xuất đặc trưng của Face CNN
            for param in self.face_features.parameters():
                param.requires_grad = False
            
            # Đóng băng toàn bộ Body Extractor
            for param in self.body_extractor.parameters():
                param.requires_grad = False

            # Đóng băng toàn bộ Context Extractor
            for param in self.context_extractor.parameters():
                param.requires_grad = False

    def forward(self, face_img, body_img, context_img):
        # Trích xuất đặc trưng từ Face-CNN
        face_map = self.face_features(face_img)
        face_feat = self.face_classifier_head(face_map)
        
        # Trích xuất đặc trưng từ Body và Context backbone
        body_feat = self.body_extractor(body_img)
        context_feat = self.context_extractor(context_img)
        
        return face_feat, body_feat, context_feat
# ==============================================================================
# 3. MÔ HÌNH KẾT HỢP
# ==============================================================================

class FusionNetwork(nn.Module):
    """Hợp nhất đặc trưng từ 3 luồng với kích thước động."""
    def __init__(self, face_dim, body_dim, context_dim, num_classes=7, 
                 projection_dim=256, hidden_dim=512,
                 use_face=True, use_body=True, use_context=True):
        super().__init__()
        self.use_face, self.use_body, self.use_context = use_face, use_body, use_context
        
        self.body_proj = nn.Linear(body_dim, projection_dim)
        self.context_proj = nn.Linear(context_dim, projection_dim)
        
        self.face_bn = nn.BatchNorm1d(face_dim)
        self.body_bn = nn.BatchNorm1d(projection_dim)
        self.context_bn = nn.BatchNorm1d(projection_dim)
        
        combined_dim = face_dim + projection_dim + projection_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, face_feat, body_feat, context_feat):
        body_proj = self.body_proj(body_feat)
        context_proj = self.context_proj(context_feat)
        
        face_bn = self.face_bn(face_feat)
        body_bn = self.body_bn(body_proj)
        context_bn = self.context_bn(context_proj)

        if not self.use_face: face_bn = torch.zeros_like(face_bn)
        if not self.use_body: body_bn = torch.zeros_like(body_bn)
        if not self.use_context: context_bn = torch.zeros_like(context_bn)

        
        
        combined_features = torch.cat([face_bn, body_bn, context_bn], dim=1)
        return self.classifier(combined_features)
    
class FusionNet_Att(nn.Module):
    def __init__(self, face_dim, body_dim, context_dim, num_classes=7, 
                 projection_dim=256, hidden_dim=512,
                 use_face=True, use_body=True, use_context=True):
        super().__init__()
        self.use_face, self.use_body, self.use_context = use_face, use_body, use_context
        
        self.face_proj = nn.Linear(face_dim, projection_dim)
        self.body_proj = nn.Linear(body_dim, projection_dim)
        self.context_proj = nn.Linear(context_dim, projection_dim)
        
        self.face_bn = nn.BatchNorm1d(projection_dim)
        self.body_bn = nn.BatchNorm1d(projection_dim)
        self.context_bn = nn.BatchNorm1d(projection_dim)

        self.modality_gating_mlp = nn.Sequential(
            nn.Linear(projection_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.cross_attention_softmax = nn.Softmax(dim=1)
        self.modality_softmax = nn.Softmax(dim=1)
        
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim*3, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(hidden_dim, num_classes)
        )

    def _perform_cross_attention(self, feat_q, feat_k1, feat_k2):
        """
        Hàm thực hiện cross-attention cho một luồng đặc trưng.
        Args:
            feat_q (Tensor): Đặc trưng truy vấn (query) - luồng cần được attention.
                             Shape: (batch, proj_dim)
            feat_k1 (Tensor): Đặc trưng khóa (key) 1. Shape: (batch, proj_dim)
            feat_k2 (Tensor): Đặc trưng khóa (key) 2. Shape: (batch, proj_dim)
        Returns:
            Tensor: Đặc trưng đã được attention. Shape: (batch, proj_dim)
        """
        # Reshape để thực hiện tích ma trận: (batch, proj_dim, 1)
        q = feat_q.unsqueeze(2)
        k1 = feat_k1.unsqueeze(2)
        k2 = feat_k2.unsqueeze(2)

        # Tính toán độ tương đồng: (batch, 1, 1)
        attention_score1 = torch.matmul(q.transpose(1, 2), k1)
        attention_score2 = torch.matmul(q.transpose(1, 2), k2)

        # Nối các điểm attention và áp dụng softmax: (batch, 2, 1)
        attention_weights = self.cross_attention_softmax(
            torch.cat((attention_score1, attention_score2), dim=1)
        )

        # Áp dụng attention weights lên các đặc trưng khóa
        # (batch, proj_dim, 1) * (batch, 1, 1) -> (batch, proj_dim, 1)
        weighted_k1 = k1 * attention_weights[:, 0, :].unsqueeze(1)
        weighted_k2 = k2 * attention_weights[:, 1, :].unsqueeze(1)
        
        # Kết hợp các đặc trưng đã được attention và cộng với đặc trưng gốc
        attended_feat = feat_q + weighted_k1.squeeze(2) + weighted_k2.squeeze(2)
        return attended_feat



    def forward(self, face_feat, body_feat, context_feat):

        face_proj = self.face_proj(face_feat)
        body_proj = self.body_proj(body_feat)
        context_proj = self.context_proj(context_feat)
        
        face_bn = self.face_bn(face_proj)
        body_bn = self.body_bn(body_proj)
        context_bn = self.context_bn(context_proj)




        if not self.use_face: face_bn = torch.zeros_like(face_bn)
        if not self.use_body: body_bn = torch.zeros_like(body_bn)
        if not self.use_context: context_bn = torch.zeros_like(context_bn)


               # ========== Bước 2: Cross-Attention giữa các phương thức ==========
        # Mỗi đặc trưng được điều chỉnh dựa trên sự tương tác với 2 đặc trưng còn lại
        attended_context = self._perform_cross_attention(context_bn, body_bn, face_bn)
        attended_body = self._perform_cross_attention(body_bn, context_bn, face_bn)
        attended_face = self._perform_cross_attention(face_bn, context_bn, body_bn)

        # ========== Bước 3: Modality Gating (Attention theo từng luồng) ==========
        # Tính một điểm số duy nhất cho mỗi luồng đặc trưng
        score_context = self.modality_gating_mlp(attended_context)
        score_body = self.modality_gating_mlp(attended_body)
        score_face = self.modality_gating_mlp(attended_face)

        # Nối các điểm số và áp dụng softmax để có trọng số
        # Shape: (batch_size, 3)
        modality_scores = torch.cat((score_context, score_body, score_face), dim=1)
        modality_weights = self.modality_softmax(modality_scores)

        # Nhân mỗi luồng đặc trưng với trọng số tương ứng của nó
        # unsqueeze(1) để broadcasting: (batch, 1) -> (batch, proj_dim)
        gated_context = attended_context * modality_weights[:, 0].unsqueeze(1)
        gated_body = attended_body * modality_weights[:, 1].unsqueeze(1)
        gated_face = attended_face * modality_weights[:, 2].unsqueeze(1)




        # Tính attention weights
        attention_weights = torch.sigmoid(gated_face + gated_body + gated_context)
        
        # Nhân các đặc trưng với attention weights
        face_att = face_bn * attention_weights
        body_att = body_bn * attention_weights
        context_att = context_bn * attention_weights
        
        combined_features = torch.cat([face_att, body_att, context_att], dim=1)
        return self.classifier(combined_features)

class CAERSNet(BaseModel):
    """Mô hình tổng thể, cho phép chọn backbone và cấu hình một cách linh hoạt."""
    def __init__(self, face_model_path, num_classes=7, 
                 body_backbone='swin_t', context_backbone='resnet50_places',
                 use_face=True, use_body=True, use_context=True,
                 projection_dim=256, hidden_dim=512, freeze_backbones=True, attention=True):
        super().__init__()
        # Truyền cấu hình backbone xuống FeatureExtractors
        self.backbone = FeatureExtractors(
            face_model_path=face_model_path, 
            num_classes=num_classes, 
            body_backbone=body_backbone, 
            context_backbone=context_backbone,
            freeze_backbones= freeze_backbones  # Đóng băng các backbone theo mặc định
        )
        
        # Lấy kích thước đặc trưng từ backbone và truyền vào FusionNetwork
        if attention:
            self.fusion_net = FusionNet_Att(
                face_dim=self.backbone.face_dim,
                body_dim=self.backbone.body_dim,
                context_dim=self.backbone.context_dim,
                num_classes=num_classes,
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                use_face=use_face,
                use_body=use_body,
                use_context=use_context
            )
        else:
            # Sử dụng FusionNetwork nếu không dùng attention
            print("Sử dụng FusionNetwork không có attention.")
            self.fusion_net = FusionNetwork(
                face_dim=self.backbone.face_dim,
                body_dim=self.backbone.body_dim,
                context_dim=self.backbone.context_dim,
                num_classes=num_classes,
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                use_face=use_face,
                use_body=use_body,
                use_context=use_context
            )

    def forward(self, face_img, body_img, context_img):
        face_feat, body_feat, context_feat = self.backbone(face_img, body_img, context_img)
        return self.fusion_net(face_feat, body_feat, context_feat)