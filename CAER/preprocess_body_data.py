import os
import argparse
import torch
from ultralytics import YOLO
from tqdm import tqdm
import shutil 
from parse_config import ConfigParser

YOLO_MODEL_NAME = 'yolov8n.pt'

def find_body_for_face(face_box, yolo_results):
    """Tìm body cho face tương ứng."""
    fx1, fy1, fx2, fy2 = face_box
    
    detected_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    detected_classes = yolo_results[0].boxes.cls.cpu().numpy()

    for i, box in enumerate(detected_boxes):
        if int(detected_classes[i]) == 0: # Lớp 'person'
            bx1, by1, bx2, by2 = box
            if bx1 <= fx1 and by1 <= fy1 and bx2 >= fx2 and by2 >= fy2:
                return tuple(map(int, box))
                
    return None

def process_dataset(root_dir, input_file, model, logger):
    """
    Hàm xử lý một file txt, phát hiện cơ thể và ghi ra file mới.
    Hàm này sẽ bỏ qua các ảnh đã được xử lý và tự động lưu tiến trình sau mỗi 100 dòng mới.
    """
    if not os.path.exists(input_file):
        logger.warning(f"File {input_file} không tồn tại. Bỏ qua.")
        return

    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_with_body{ext}"
    
    logger.info(f"Đang xử lý: {input_file}")
    logger.info(f"File output sẽ được lưu tại: {output_file}")

    # <<< SỬA ĐỔI 1: Đọc các ảnh đã xử lý để không làm lại >>>
    processed_images = set()
    if os.path.exists(output_file):
        logger.info(f"Phát hiện file đã xử lý. Đang đọc kết quả cũ để bỏ qua...")
        with open(output_file, 'r') as f_old:
            for line in f_old:
                try:
                    # Giả sử đường dẫn ảnh luôn là phần tử đầu tiên
                    image_path = line.strip().split(',')[0]
                    processed_images.add(image_path)
                except IndexError:
                    continue # Bỏ qua các dòng trống hoặc không hợp lệ
    
    logger.info(f"Đã tìm thấy {len(processed_images)} ảnh đã được xử lý. Sẽ bỏ qua chúng.")

    # <<< SỬA ĐỔI 2: Mở file output ở chế độ 'append' (ghi tiếp) >>>
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
            lines_to_process = f_in.readlines()
            
            # Biến đếm để lưu sau mỗi 100 dòng mới
            new_lines_count = 0 
            
            progress_bar = tqdm(lines_to_process, desc=f"Processing {os.path.basename(input_file)}")

            for line in progress_bar:
                try:
                    line_strip = line.strip()
                    parts = line_strip.split(',')
                    
                    if not parts or not parts[0]:
                        continue

                    relative_img_path = parts[0]

                    # <<< SỬA ĐỔI 3: KIỂM TRA VÀ BỎ QUA NẾU ĐÃ XỬ LÝ >>>
                    if relative_img_path in processed_images:
                        continue 

                    # Nếu chưa xử lý, tiến hành chạy model
                    if len(parts) < 6:
                        # Ghi lại dòng không hợp lệ vào file output để không kiểm tra lại
                        f_out.write(line_strip + '\n')
                        logger.warning(f"Bỏ qua dòng không hợp lệ nhưng đã ghi lại: {line_strip}")
                        continue

                    face_coords = tuple(map(int, parts[2:6]))
                    full_img_path = os.path.join(root_dir, relative_img_path)
                    
                    if not os.path.exists(full_img_path):
                        logger.warning(f"Không tìm thấy file ảnh: {full_img_path}. Bỏ qua.")
                        continue

                    results = model(full_img_path, verbose=False)
                    body_coords = find_body_for_face(face_coords, results)
                    
                    if body_coords is None:
                        body_coords = (0, 0, 0, 0)
                        
                    new_line_parts = parts[:6] + [str(c) for c in body_coords]
                    f_out.write(','.join(new_line_parts) + '\n')
                    
                    # <<< SỬA ĐỔI 4: LƯU FILE SAU MỖI 100 DÒNG MỚI >>>
                    new_lines_count += 1
                    if new_lines_count % 100 == 0:
                        f_out.flush()  # Đẩy dữ liệu từ buffer vào file hệ thống
                        os.fsync(f_out.fileno()) # Ép hệ thống ghi file xuống đĩa
                        progress_bar.set_postfix_str(f"Đã lưu! (Thêm {new_lines_count} dòng mới)")

                except Exception as e:
                    logger.error(f"Lỗi khi xử lý dòng '{line.strip()}': {e}")
            
            # Lưu lần cuối để đảm bảo mọi thứ được ghi lại
            f_out.flush()
            os.fsync(f_out.fileno())

        logger.info(f"Hoàn tất xử lý và đã cập nhật tại: {output_file}")

    except IOError as e:
        logger.error(f"Không thể đọc/ghi file: {e}")
    except Exception as e:
        logger.error(f"Đã xảy ra lỗi không mong muốn: {e}")

# <<< SỬA ĐỔI 1: Hàm main nhận 'device' làm tham số >>>
def main(config, device):
    logger = config.get_logger('data_processing')
    logger.info("Bắt đầu quá trình tiền xử lý dữ liệu để thêm tọa độ cơ thể.")
    
    # <<< SỬA ĐỔI 2: Sử dụng trực tiếp 'device' đã được truyền vào >>>
    logger.info(f"Sử dụng thiết bị: {device}")

    logger.info(f"Đang tải model YOLO: {YOLO_MODEL_NAME}...")
    model = YOLO(YOLO_MODEL_NAME)
    model.to(device)
    logger.info("Tải model thành công.")

    # Lấy thông tin từ config
    datasets_to_process = {
        'Train': (config['train_loader']['args']['root'], config['train_loader']['args']['detect_file']),
        'Validation': (config['val_loader']['args']['root'], config['val_loader']['args']['detect_file']),
        'Test': (config['test_loader']['args']['root'], config['test_loader']['args']['detect_file'])
    }

    # Xử lý từng tập dữ liệu
    for name, (root, filepath) in datasets_to_process.items():
        logger.info(f"===== Bắt đầu xử lý tập {name} =====")
        process_dataset(root, filepath, model, logger)
        logger.info(f"===== Kết thúc xử lý tập {name} =====")
    
    logger.info("Toàn bộ quá trình tiền xử lý đã hoàn tất.")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Pytorch Template Preprocessing')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (e.g. 0,1,2), default: auto-detect')
    # Thêm lại resume vì from_args có thể cần nó
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # <<< SỬA ĐỔI QUAN TRỌNG >>>
    # Sử dụng from_args() để tạo đối tượng config
    config = ConfigParser.from_args(args)
    
    # Logic chuẩn bị device
    parsed_args = args.parse_args() # Phải gọi lại parse_args để lấy giá trị device
    if parsed_args.device is not None:
        device = torch.device(f'cuda:{parsed_args.device}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(config, device)