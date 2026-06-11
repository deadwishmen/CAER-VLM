import os
import argparse
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import textwrap

def visualize_samples(root_dir, input_file, output_dir, num_samples=10):
    """
    Đọc file dữ liệu, vẽ bounding box đỏ cho khuôn mặt và hiển thị cùng văn bản mô tả.
    """
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file '{input_file}'.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("File dữ liệu trống.")
        return

    # Chọn ngẫu nhiên các mẫu để trực quan hóa
    selected_lines = random.sample(lines, min(num_samples, len(lines)))

    print(f"Đang xử lý {len(selected_lines)} mẫu...")

    for i, line in enumerate(selected_lines):
        parts = line.split(',')
        if len(parts) < 11:
            print(f"Bỏ qua dòng không đủ dữ liệu: {line[:50]}...")
            continue

        image_path = parts[0]
        full_image_path = os.path.join(root_dir, image_path)

        if not os.path.exists(full_image_path):
            print(f"Không tìm thấy ảnh: {full_image_path}")
            continue

        try:
            # Tọa độ khuôn mặt (Cột 3, 4, 5, 6 trong tệp annotation)
            x1_face, y1_face, x2_face, y2_face = map(int, parts[2:6])
            
            # Lấy phần mô tả (bắt đầu từ cột thứ 11)
            description = ','.join(parts[10:]).strip()
            
            # Xóa tiền tố ASSISTANT: nếu có
            if description.startswith("ASSISTANT:"):
                description = description[len("ASSISTANT:"):].strip()
            
            # Đọc ảnh và vẽ box
            img = Image.open(full_image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Vẽ box đỏ cho face (đối tượng)
            draw.rectangle([x1_face, y1_face, x2_face, y2_face], outline="red", width=4)
            
            # Thiết lập biểu đồ với matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Ảnh: {os.path.basename(image_path)}", fontsize=14, fontweight='bold')
            
            # Tự động xuống dòng cho văn bản mô tả để không bị tràn
            wrapped_text = textwrap.fill(description, width=90)
            
            # Hiển thị văn bản phía dưới ảnh với phông nền vàng nhạt để dễ đọc
            plt.figtext(0.5, 0.0, wrapped_text, wrap=True, horizontalalignment='center', 
                        fontsize=12, bbox={"facecolor": "lightyellow", "alpha": 0.8, "pad": 8, "edgecolor": "gray"})
            
            # Lưu ảnh
            save_path = os.path.join(output_dir, f"visualized_sample_{i+1}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
            plt.close(fig)
            
            print(f"Đã lưu: {save_path}")

        except Exception as e:
            print(f"Lỗi khi xử lý {image_path}: {e}")

    print(f"Hoàn tất! Các ảnh đã được lưu tại thư mục: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trực quan hóa ảnh kèm bounding box đỏ và văn bản mô tả.")
    parser.add_argument('--root', type=str, required=True, help="Thư mục gốc chứa ảnh (Root Directory).")
    parser.add_argument('--input_file', type=str, required=True, help="File txt kết quả từ quá trình image2text (đã có đoạn văn bản).")
    parser.add_argument('--output_dir', type=str, default='visualizations', help="Thư mục lưu các ảnh sau khi ghép (Mặc định: visualizations).")
    parser.add_argument('--num_samples', type=int, default=10, help="Số lượng mẫu ngẫu nhiên muốn xử lý.")
    
    args = parser.parse_args()
    visualize_samples(args.root, args.input_file, args.output_dir, args.num_samples)