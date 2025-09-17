from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import DataLoader, Dataset
import utils.util as ut 
from PIL import Image, ImageDraw
import os
from transformers import ViltProcessor, ViltModel, ViltConfig

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

def load_vilt_processor(vilt_model_name="dandelin/vilt-b32-mlm", cache_dir=None):
  try:
      offline_mode = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
      processor = ViltProcessor.from_pretrained(
          vilt_model_name, local_files_only=offline_mode, cache_dir=cache_dir
      )
      print(f"Loaded ViltProcessor from {vilt_model_name} successfully.")
      return processor
  except Exception as e:
      print(f"ERROR: Failed to load ViltProcessor: {e}")
      raise

class MyDataset(Dataset):
    """
    Lớp Dataset được tối ưu hóa:
    - Phân tích và làm sạch toàn bộ dữ liệu annotation một lần duy nhất khi khởi tạo.
    - __getitem__ chỉ tập trung vào việc tải và xử lý ảnh, giúp tăng tốc độ.
    - Xử lý lỗi chi tiết và mạnh mẽ hơn.
    """
    def __init__(self, root, input_file, transform=None, default_body_size=(224, 224), text_max_len = 256):
        self.root = root
        self.transform = transform
        self.default_body_size = default_body_size
        self.vilt_processor = load_vilt_processor()
        self.text_max_len = text_max_len
        


        # Phân tích toàn bộ file và lưu kết quả đã được làm sạch
        self.samples = self._read_and_parse_input_file(input_file)
        if not self.samples:
          
            print("Cảnh báo: Không có mẫu dữ liệu hợp lệ nào được tải.")

    def _read_and_parse_input_file(self, file_path):
        """Đọc và phân tích file annotation, lọc ra các dòng bị lỗi."""
        parsed_samples = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Lỗi nghiêm trọng: Không tìm thấy file annotation tại '{file_path}'")
            return []

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')

            coords = []
            for p in parts[2:9]:
                p = p.strip()
                if not p:
                    print(f"Cảnh báo: Bỏ qua dòng {idx+1} do giá trị coords rỗng: '{line}'")
                    coords.append(int(p))


            description_parts = parts[10:]
            description = [p.strip() for p in description_parts if p.strip()]
            if description and description[0].startswith("ASSISTANT:"):
              description[0] = description[0][len("ASSISTANT:"):].strip()
              # Loại bỏ phần tử rỗng sau khi bỏ "ASSISTANT:" nếu cần
              description = [p for p in description if p]
            try:
                # Đóng gói dữ liệu đã phân tích thành một dictionary
                sample_data = {
                    "image_path": parts[0],
                    "label": int(parts[1]),
                    "coords": coords,
                    "description": description
                }
                parsed_samples.append(sample_data)
            except ValueError:
                print(f"Cảnh báo: Bỏ qua dòng {idx+1} do lỗi chuyển đổi sang số nguyên: '{line}'")
                continue
        


        return parsed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Lấy thông tin đã được phân tích sẵn
        sample_info = self.samples[idx]
        


        try:
            # 1. Tải ảnh
            full_path = os.path.join(self.root, sample_info["image_path"])
            im = Image.open(full_path).convert('RGB')

            # 2. Xử lý tọa độ và cắt ảnh
            coords = sample_info["coords"]
            x1_face, y1_face, x2_face, y2_face, x1_body, y1_body, x2_body, y2_body = coords
            
            face = im.crop((x1_face, y1_face, x2_face, y2_face))

            if all(c == 0 for c in coords[4:]):
                body = Image.new('RGB', self.default_body_size, (0, 0, 0))
            else:
                body = im.crop((x1_body, y1_body, x2_body, y2_body))

            # 3. Tạo context
            context = im.copy()
            draw = ImageDraw.Draw(context)
            draw.rectangle((x1_face, y1_face, x2_face, y2_face), fill=(0, 0, 0))

            text = sample_info["description"]  


            inputs_context = self.processor(
                images=context, text=text, return_tensors="pt",
                padding="max_length", truncation=True, max_length=self.text_max_len
            )

            inputs_face = self.processor(
                images=face, text=text, return_tensors="pt",
                padding="max_length", truncation=True, max_length=self.text_max_len
            )
            

            # 4. Đóng gói và áp dụng transform
            data_dict = {'input_ids': inputs_context['input_ids'].squeeze(0),
                        'attention_mask': inputs_context['attention_mask'].squeeze(0),
                        'token_type_ids': inputs_context['token_type_ids'].squeeze(0),
                        'pixel_values_context': inputs_context['pixel_values'].squeeze(0),
                        'token_type_ids_context': inputs_context['token_type_ids'].squeeze(0),
                        'pixel_values_face': inputs_face['pixel_values'].squeeze(0),
                        'token_type_ids_face': inputs_face['token_type_ids'].squeeze(0)}
            
            if self.transform:
                data_dict = self.transform(data_dict)

            return data_dict, sample_info["label"]

        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file ảnh tại '{full_path}'. Bỏ qua mẫu này.")
            return None
        except Exception as e:
            # Bắt các lỗi khác liên quan đến xử lý ảnh (ví dụ: ảnh bị hỏng)
            print(f"Lỗi khi xử lý ảnh '{full_path}'. Lỗi: {e}. Bỏ qua mẫu này.")
            return None

class CAERSDataLoader(BaseDataLoader):
    def __init__(self, root, detect_file, train=True, batch_size=32, shuffle=True, num_workers=2):
        """
        Create dataloader from directory
        Args:
            - root (str): root directory
            - detect_file (str): file containing results from detector 
        """
        
        # data_transforms = ut.get_transform(train)
        self.dataset = MyDataset(root, detect_file, transform=None)
        super().__init__(self.dataset, batch_size, shuffle, validation_split=0.0, num_workers=num_workers, collate_fn=collate_fn)
