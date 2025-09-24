from torchvision import datasets
from base import BaseDataLoader
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import os
from transformers import ViltProcessor, ViltModel, ViltConfig

def collate_fn(batch):
    # Lọc các mẫu None
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None  # Trả về None nếu batch rỗng

    # Tách data_dicts và labels
    data_dicts, labels = zip(*batch)

    # Tạo dictionary mới để lưu batch
    batched_dict = {}
    
    # Xếp chồng các tensor
    tensor_keys = ['input_ids', 'attention_mask', 'token_type_ids', 
                   'pixel_values_context', 'pixel_values_face']
    for key in tensor_keys:
        batched_dict[key] = torch.utils.data.dataloader.default_collate([d[key] for d in data_dicts])
    
    # Giữ ảnh PIL dưới dạng list
    pil_keys = ['face', 'body', 'context']
    for key in pil_keys:
        batched_dict[key] = [d[key] for d in data_dicts]
    
    # Xếp chồng labels
    labels = torch.utils.data.dataloader.default_collate(labels)
    
    return batched_dict, labels

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
    def __init__(self, root, input_file, default_body_size=(224, 224), text_max_len=256, image_size=(384, 384)):
        self.root = root
        self.default_body_size = default_body_size
        self.vilt_processor = load_vilt_processor()
        self.text_max_len = text_max_len
        self.image_size = image_size  # Kích thước cố định cho ảnh

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
            if len(parts) < 10:
                print(f"Cảnh báo: Bỏ qua dòng {idx+1} do thiếu cột (cần ít nhất 10 cột): '{line}'")
                continue

            try:
                # Lấy tọa độ và chuyển đổi sang số nguyên
                coords = [int(p) for p in parts[2:10]]
                if len(coords) != 8:
                    print(f"Cảnh báo: Bỏ qua dòng {idx+1} do thiếu tọa độ: '{coords}'")
                    continue

                # Kiểm tra tính hợp lệ của tọa độ
                x1_face, y1_face, x2_face, y2_face, x1_body, y1_body, x2_body, y2_body = coords
                if x1_face >= x2_face or y1_face >= y2_face:
                    print(f"Cảnh báo: Bỏ qua dòng {idx+1} do tọa độ khuôn mặt không hợp lệ: ({x1_face}, {y1_face}, {x2_face}, {y2_face})")
                    continue
                if not all(c == 0 for c in coords[4:]) and (x1_body >= x2_body or y1_body >= y2_body):
                    print(f"Cảnh báo: Bỏ qua dòng {idx+1} do tọa độ cơ thể không hợp lệ: ({x1_body}, {y1_body}, {x2_body}, {y2_body})")
                    continue

                # Xử lý phần mô tả
                description = ','.join(parts[10:]).strip()  # Nối lại các phần tử mô tả
                if description.startswith("ASSISTANT:"):
                    description = description[len("ASSISTANT:"):].strip()

                # Đóng gói dữ liệu
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
        sample_info = self.samples[idx]

        try:
            # 1. Tải ảnh
            full_path = os.path.join(self.root, sample_info["image_path"])
            im = Image.open(full_path).convert('RGB')
            img_width, img_height = im.size

            # 2. Kiểm tra tọa độ so với kích thước ảnh
            coords = sample_info["coords"]
            x1_face, y1_face, x2_face, y2_face, x1_body, y1_body, x2_body, y2_body = coords
            for coord, name in [(x1_face, 'x1_face'), (x2_face, 'x2_face'), (x1_body, 'x1_body'), (x2_body, 'x2_body')]:
                if coord < 0 or coord > img_width:
                    print(f"Lỗi: Tọa độ {name} ({coord}) ngoài giới hạn chiều rộng ảnh ({img_width}) tại '{full_path}'")
                    return None
            for coord, name in [(y1_face, 'y1_face'), (y2_face, 'y2_face'), (y1_body, 'y1_body'), (y2_body, 'y2_body')]:
                if coord < 0 or coord > img_height:
                    print(f"Lỗi: Tọa độ {name} ({coord}) ngoài giới hạn chiều cao ảnh ({img_height}) tại '{full_path}'")
                    return None

            # 3. Cắt ảnh
            face = im.crop((x1_face, y1_face, x2_face, y2_face))
            face = face.resize(self.image_size, Image.Resampling.LANCZOS)  # Thay đổi kích thước
            if all(c == 0 for c in coords[4:]):
                body = Image.new('RGB', self.default_body_size, (0, 0, 0))
            else:
                body = im.crop((x1_body, y1_body, x2_body, y2_body))
                body = body.resize(self.image_size, Image.Resampling.LANCZOS)  # Thay đổi kích thước

            # 4. Tạo context
            context = im.copy()
            draw = ImageDraw.Draw(context)
            draw.rectangle((x1_face, y1_face, x2_face, y2_face), outline="red", width=3)
            context = context.resize(self.image_size, Image.Resampling.LANCZOS)  # Thay đổi kích thước

            # 5. Xử lý với ViltProcessor
            text = sample_info["description"]
            inputs_context = self.vilt_processor(
                images=context, text=text, return_tensors="pt",
                padding="max_length", truncation=True, max_length=self.text_max_len
            )
            inputs_face = self.vilt_processor(
                images=face, text=text, return_tensors="pt",
                padding="max_length", truncation=True, max_length=self.text_max_len
            )

            # 6. Đóng gói
            data_dict = {
                'input_ids': inputs_context['input_ids'].squeeze(0),
                'attention_mask': inputs_context['attention_mask'].squeeze(0),
                'token_type_ids': inputs_context['token_type_ids'].squeeze(0),
                'pixel_values_context': inputs_context['pixel_values'].squeeze(0),
                'pixel_values_face': inputs_face['pixel_values'].squeeze(0),
                'face': face,  # Giữ ảnh PIL cho TensorBoard
                'body': body,
                'context': context
            }

            return data_dict, sample_info["label"]

        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file ảnh tại '{full_path}'. Bỏ qua mẫu này.")
            return None
        except Exception as e:
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
        self.dataset = MyDataset(root, detect_file)
        super().__init__(self.dataset, batch_size, shuffle, validation_split=0.0, num_workers=num_workers, collate_fn=collate_fn)