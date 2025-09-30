import json
import numpy as np 
import pandas as pd
from pathlib import Path
from PIL import Image
from itertools import repeat
from collections import OrderedDict
import torchvision 
from torchvision import transforms
import torchvision.datasets as dset
import torch 
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

def get_path_images(root, test_size=0.0, mode=0):
    """
    Lấy đường dẫn ảnh để tạo dataset
    Args:
    - root: thư mục chứa ảnh
    - test_size: tỷ lệ tập validation
    - mode: 0 (toàn bộ dataset), n (1024*n ảnh)
    """
    img_folder = ImageFolder(root)
    train = img_folder.imgs
    np.random.seed(42)
    np.random.shuffle(train)

    if mode != 0: # Dùng để debug
        train = train[:1024*mode]
    class_to_idx = img_folder.class_to_idx

    if test_size > 0.0:
        num_train = int((1-test_size)*len(train))
        print('Trainset size: {}, Valset size: {}'.format(num_train, len(train)-num_train))
        return (train[:num_train], train[num_train:]), class_to_idx
    print('Testset size:', len(train))
    return train, class_to_idx

# ==============================================================================
# ĐỊNH NGHĨA CÁC LỚP TRANSFORM TÙY CHỈNH
# ==============================================================================

# Thay thế lớp Resize cũ bằng lớp này trong file data_loaders.py của bạn

class Resize(object):
    """
    Resize các vùng mặt, cơ thể và ngữ cảnh với các kích thước khác nhau.
    Đã sửa lỗi để đảm bảo kích thước đầu ra luôn cố định.
    """
    def __init__(self, sizes):
        # sizes là một tuple chứa 3 kích thước: (face_size, body_size, context_size)
        assert isinstance(sizes, tuple) and len(sizes) == 3
        
        face_size, body_size, context_size = sizes

        # Đảm bảo các kích thước đều là tuple (H, W) để ép ảnh về đúng size
        if isinstance(face_size, int):
            face_size = (face_size, face_size)
        if isinstance(body_size, int):
            body_size = (body_size, body_size)
        if isinstance(context_size, int):
            context_size = (context_size, context_size)

        self.face_resize = transforms.Resize(face_size)
        self.body_resize = transforms.Resize(body_size)
        self.context_resize = transforms.Resize(context_size)

    def __call__(self, sample):
        # sample là một dictionary: {'face': img, 'body': img, 'context': img}
        face = sample['face']
        body = sample['body']
        context = sample['context']
        
        return {
            'face': self.face_resize(face),
            'body': self.body_resize(body),
            'context': self.context_resize(context)
        }

class Crop(object):
    """
    Cắt (ngẫu nhiên hoặc trung tâm) vùng ngữ cảnh.
    """
    def __init__(self, size, mode="train"):
        if mode == "train":
            self.cropper = transforms.RandomCrop(size)
        else:
            self.cropper = transforms.CenterCrop(size)

    def __call__(self, sample):
        # Chỉ cắt ảnh context, giữ nguyên face và body
        sample['context'] = self.cropper(sample['context'])
        return sample

class ToTensorAndNormalize(object):
    """
    Chuyển ảnh PIL thành Tensor và chuẩn hóa với số kênh phù hợp.
    """
    def __init__(self):
        # Khởi tạo các phép biến đổi một lần để tối ưu
        self.to_tensor = transforms.ToTensor()
        # Chuẩn hóa cho ảnh RGB 3 kênh (body, context)
        self.norm_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Chuẩn hóa cho ảnh xám 1 kênh (face)
        self.norm_gray = transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, sample):
        # Chuyển đổi định dạng kênh ảnh cho phù hợp với từng model
        face = sample['face'].convert("L")       # 'L' là chế độ ảnh xám (1 kênh)
        body = sample['body'].convert("RGB")     # 'RGB' là ảnh màu (3 kênh)
        context = sample['context'].convert("RGB")
        
        # Áp dụng ToTensor và Normalize tương ứng
        face_tensor = self.norm_gray(self.to_tensor(face))
        body_tensor = self.norm_rgb(self.to_tensor(body))
        context_tensor = self.norm_rgb(self.to_tensor(context))
        
        # Thêm một chiều batch giả cho face tensor để có dạng [1, 1, H, W]
        # nếu cần, nhưng thường DataLoader sẽ tự xử lý
        return {
            'face': face_tensor,
            'body': body_tensor,
            'context': context_tensor,
        }

# ==============================================================================
# HÀM TẠO CHUỖI TRANSFORM
# ==============================================================================

def get_transform(train=True):
    """
    Tạo và trả về một chuỗi các phép biến đổi cho training hoặc validation/test.
    """
    # Kích thước resize: face=96, body=224, context=256 (lớn hơn crop size)
    resize_sizes = (48, 224, 224)
    # Kích thước crop cho context
    crop_size = 224
    
    # Chọn chế độ crop
    crop_mode = "train" if train else "test"
    
    return transforms.Compose([
        Resize(resize_sizes),
        Crop(crop_size, mode=crop_mode),
        ToTensorAndNormalize()
    ])



def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
