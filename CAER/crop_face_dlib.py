import os
import argparse
import cv2
import dlib
from tqdm import tqdm
import urllib.request
import bz2
from parse_config import ConfigParser

def download_dlib_model(model_path):
    """
    Tải pretrained model Dlib CNN Face Detector nếu chưa tồn tại trong thư mục hiện tại.
    """
    if not os.path.exists(model_path):
        print(f"Đang tải model Dlib CNN từ internet vào {model_path}...")
        url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
        bz2_path = model_path + ".bz2"
        urllib.request.urlretrieve(url, bz2_path)
        
        print(f"Đang giải nén model...")
        with bz2.BZ2File(bz2_path, 'rb') as fr, open(model_path, 'wb') as fw:
            fw.write(fr.read())
        os.remove(bz2_path)
        print("Tải model thành công!")

def process_dataset(root_dir, input_file, cnn_face_detector, logger):
    """
    Quét qua các file ảnh, nhận diện mặt bằng Dlib CNN và cập nhật tọa độ.
    """
    if not os.path.exists(input_file):
        logger.warning(f"File {input_file} không tồn tại. Bỏ qua.")
        return

    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_dlib_face{ext}"
    
    logger.info(f"Đang xử lý: {input_file}")
    logger.info(f"File output sẽ được lưu tại: {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        for line in tqdm(lines, desc=f"Processing {os.path.basename(input_file)}"):
            line_strip = line.strip()
            if not line_strip:
                continue
                
            parts = line_strip.split(',')
            # File annotation của CAER cần ít nhất 6 cột để chứa tọa độ mặt
            if len(parts) < 6:
                f_out.write(line_strip + '\n')
                continue
                
            relative_img_path = parts[0]
            full_img_path = os.path.join(root_dir, relative_img_path)
            
            if not os.path.exists(full_img_path):
                logger.warning(f"Không tìm thấy file ảnh: {full_img_path}")
                f_out.write(line_strip + '\n')
                continue

            # Đọc ảnh bằng cv2
            img = cv2.imread(full_img_path)
            if img is None:
                f_out.write(line_strip + '\n')
                continue
            
            # Dlib hoạt động tốt nhất với ảnh RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Nhận diện khuôn mặt với tham số upsample = 1
            dets = cnn_face_detector(img_rgb, 1)
            
            if len(dets) > 0:
                # Nếu có nhiều khuôn mặt, chọn cái có confidence score cao nhất
                dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
                face_rect = dets[0].rect
                
                # Cắt tọa độ đảm bảo không vượt quá kích thước ảnh
                x1 = max(0, face_rect.left())
                y1 = max(0, face_rect.top())
                x2 = min(img.shape[1], face_rect.right())
                y2 = min(img.shape[0], face_rect.bottom())
                
                # Thay thế box cắt khuôn mặt cũ bằng box mới
                parts[2] = str(x1)
                parts[3] = str(y1)
                parts[4] = str(x2)
                parts[5] = str(y2)
            
            # Ghi dòng dữ liệu ra file output (dùng tọa độ mới hoặc giữ nguyên tọa độ cũ nếu ko có mặt)
            f_out.write(','.join(parts) + '\n')

def main(config):
    logger = config.get_logger('dlib_face_crop')
    
    model_path = "mmod_human_face_detector.dat"
    download_dlib_model(model_path)
    
    logger.info("Đang khởi tạo Dlib CNN Face Detector...")
    # Khởi tạo mô hình Dlib CNN
    cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)

    datasets_to_process = {
        'Train': (config['train_loader']['args']['root'], config['train_loader']['args']['detect_file']),
        'Validation': (config['val_loader']['args']['root'], config['val_loader']['args']['detect_file']),
        'Test': (config['test_loader']['args']['root'], config['test_loader']['args']['detect_file'])
    }

    # Lặp qua từng tập dataset để cập nhật box
    for name, (root, filepath) in datasets_to_process.items():
        logger.info(f"===== Bắt đầu xử lý tập {name} =====")
        process_dataset(root, filepath, cnn_face_detector, logger)
        logger.info(f"===== Kết thúc xử lý tập {name} =====")
    
    logger.info("Hoàn tất thay thế tọa độ khuôn mặt!")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Dlib CNN Face Crop Annotation Replacement')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable')
    
    config = ConfigParser.from_args(args)
    main(config)