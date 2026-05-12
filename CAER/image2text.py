import numpy as np
import os
from tqdm import tqdm
import torch
import argparse
from parse_config import ConfigParser
from transformers import pipeline
from PIL import Image, ImageDraw
from typing import List, Tuple
import multiprocessing as mp
from multiprocessing import Lock # ### THAY ĐỔI ###: Import thêm Lock
from torch.utils.data import Dataset, DataLoader
import math

def read_image(image_path):
    """
    the functions reads an image from the given path and returns a PIL Image object
    """
    return Image.open(image_path).convert('RGB')

def get_assistant_text(text):
    start_index = text.find("ASSISTANT")
    assistant_text = text[start_index:] if start_index != -1 else ""
    return assistant_text

class ImageTextDataset(Dataset):
    def __init__(self, items):
        self.items = items
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            image = read_image(item['full_image_path'])
            x1, y1, x2, y2 = item['face_coords']
            draw = ImageDraw.Draw(image)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            return {"valid": True, "image": image, "base_info": item['base_info']}
        except Exception as e:
            return {"valid": False, "error": str(e), "item": item}

def custom_collate(batch):
    valid_batch = [b for b in batch if b["valid"]]
    if not valid_batch:
        return None
    images = [b["image"] for b in valid_batch]
    base_infos = [b["base_info"] for b in valid_batch]
    return images, base_infos

def prepare_data_items(lines: List[str], root: str, processed_images: set, logger):
    """
    Prepare list of data items for processing
    """
    items = []
    for line in lines:
        try:
            parts = line.strip().split(',')
            if not parts or not parts[0]:
                continue
            
            image_path = parts[0]
            
            if image_path in processed_images:
                continue
            
            if len(parts) >= 10:
                base_info = ','.join(parts[:10])
            else:
                base_info = ','.join(parts)

            face_coords = tuple(map(int, parts[2:6]))
            full_image_path = os.path.join(root, image_path)
            
            if not os.path.exists(full_image_path):
                logger.warning(f"Image {full_image_path} does not exist. Skipping.")
                continue
            
            items.append({
                'base_info': base_info,
                'image_path': image_path,
                'full_image_path': full_image_path,
                'face_coords': face_coords
            })
        except Exception as e:
            logger.error(f"Failed to parse line: {line.strip()}. Error: {e}")
            continue
    return items

### THAY ĐỔI ###: Sửa hàm worker để sử dụng DataLoader tối ưu I/O và bfloat16 cho A100
def process_worker(gpu_id, items_chunk, model_id, prompts, max_new_tokens, output_file, lock, config):
    """
    Hàm worker, mỗi worker sẽ chạy trên một tiến trình và một GPU riêng biệt.
    Đã tối ưu hóa DataLoader và Precision cho A100 80GB + 40GB RAM.
    """
    logger = config.get_logger(f'worker_gpu_{gpu_id}')
    device = torch.device(f'cuda:{gpu_id}')
    batch_size = config['image2text'].get('batch_size', 8)
    logger.info(f"Worker on GPU {gpu_id} started, processing {len(items_chunk)} items with batch size {batch_size}.")

    # Tối ưu cho A100: Sử dụng bfloat16 (Ampere Architecture) để tối đa hoá Tensor Cores
    # Cải thiện tốc độ và tránh lỗi tràn số so với float16
    model_kwargs = {
        "torch_dtype": torch.bfloat16, 
        "device_map": {"": device}
    }
    
    try:
        pipe = pipeline(
            "image-text-to-text", 
            model=model_id,
            model_kwargs=model_kwargs
        )
        logger.info(f"Pipeline created successfully on GPU {gpu_id}")
    except Exception as e:
        logger.error(f"Failed to create pipeline on GPU {gpu_id}: {e}")
        return

    # Khởi tạo DataLoader để xử lý tải và cắt ảnh trên CPU thông qua đa tiến trình (tận dụng 40GB RAM)
    dataset = ImageTextDataset(items_chunk)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4,          # 4 workers đủ để load ảnh nhanh chóng mà không tràn RAM
        collate_fn=custom_collate, 
        prefetch_factor=2,      # Pre-fetch 2 batch sẵn vào hàng đợi
        shuffle=False
    )

    progress_bar = tqdm(dataloader, desc=f"GPU {gpu_id}", position=gpu_id)
    for batch in progress_bar:
        if batch is None:
            continue
            
        images, base_infos = batch
        batch_prompts = [prompts] * len(images)
        
        try:
            # Đẩy qua model
            results = pipe(
                images, 
                text=batch_prompts, 
                generate_kwargs={"max_new_tokens": max_new_tokens},
                batch_size=len(images)
            )
            
            output_lines = []
            for i, result in enumerate(results):
                # Lấy text an toàn từ pipeline
                if isinstance(result, list):
                    output_prompt = result[0]["generated_text"]
                else:
                    output_prompt = result["generated_text"]
                    
                output_prompt = get_assistant_text(output_prompt)
                output_lines.append(f"{base_infos[i]},{output_prompt}")
            
            # Ghi nối vào tệp chung một cách thread-safe
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f_out:
                    for line in output_lines:
                        f_out.write(f"{line}\n")
                        
        except Exception as e:
            logger.error(f"Failed to process batch on GPU {gpu_id}. Error: {e}")
            
    logger.info(f"Worker on GPU {gpu_id} finished.")


### THAY ĐỔI ###: Sửa hàm điều phối để dùng Lock và không tạo/gộp file tạm
def run_inference_on_file(root, input_file, config, prompts, max_new_tokens, batch_size, device_ids, logger):
    """
    Hàm này điều phối việc xử lý một file trên nhiều GPU.
    """
    if not os.path.exists(input_file):
        logger.warning(f"File {input_file} does not exist. Skipping.")
        return

    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_inference{ext}"

    logger.info(f"Processing file: {input_file}")
    logger.info(f"Output will be saved to: {output_file}")
    logger.info(f"Using GPUs: {device_ids}")

    processed_images = set()
    if os.path.exists(output_file):
        logger.info("Processed file detected. Loading previous results to skip...")
        with open(output_file, 'r', encoding='utf-8') as f_old:
            for line in f_old:
                try:
                    image_path = line.strip().split(',')[0]
                    processed_images.add(image_path)
                except IndexError:
                    continue
    logger.info(f"Found {len(processed_images)} processed images. Will skip them.")

    with open(input_file, 'r', encoding='utf-8') as f_in:
        all_lines = f_in.readlines()

    logger.info("Preparing data items...")
    items = prepare_data_items(all_lines, root, processed_images, logger)
    logger.info(f"Total items to process: {len(items)}")
    
    if not items:
        logger.info("No new data to process.")
        return

    num_gpus = len(device_ids)
    
    # Chia đều số lượng items cho các GPU
    chunk_size = math.ceil(len(items) / num_gpus)
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    # Tạo Lock để chia sẻ giữa các tiến trình
    manager = mp.Manager()
    lock = manager.Lock()
    
    worker_args = []
    for i, gpu_id in enumerate(device_ids):
        if i < len(chunks) and chunks[i]:
            worker_args.append((
                gpu_id, 
                chunks[i],
                config['image2text']['model_id'],
                prompts,
                max_new_tokens,
                output_file,
                lock,
                config
            ))

    ctx = mp.get_context('spawn')
    processes = []
    for args in worker_args:
        p = ctx.Process(target=process_worker, args=args)
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    # Không cần gộp file nữa
    logger.info(f"All workers finished. Results have been saved to {output_file}.")


def main(config, device_ids, prompt_file=None):
    logger = config.get_logger('image2text')
    logger.info("Starting inference process...")
    logger.info(f"Using devices: {device_ids}")

    class_names = config['class_names']
    max_new_tokens = config['image2text']['max_new_tokens']
    
    batch_size = config['image2text'].get('batch_size', 8)
    logger.info(f"Using batch size per GPU: {batch_size}")
    
    emotions_str = ', '.join(class_names)

    prompt = (
        "USER: <image>\n"
        f"Given the following list of emotions: {emotions_str}. "
        "Based on the image context, please choose which emotions are more suitable "
        "for describing how the person in the red box feels and explain in detail "
        "why you choose these emotions according to the aspects: "
        "actions and postures of the person in the red box, "
        "facial expressions of the person in the red box "
        "(e.g., eye contact, mouth shape, eyebrow position, and overall facial tension), "
        "the context surrounding the person in the red box.\n"
        "ASSISTANT:"
    )
    if prompt_file and os.path.exists(prompt_file):
        logger.info(f"Loading prompt from file: {prompt_file}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = f.read().replace('{class_names}', ', '.join(class_names))
    else:
        prompts = f"USER: <image>\nGiven the following list of emotions: {', '.join(class_names)}. Based on the image context, please choose which emotions are more suitable for describing how the person in the red box feels and explain in detail why you choose these emotions according to the aspects: actions and postures of the person in the red box, the context surrounding the person in the red box.\nASSISTANT:"
        
    model_id = config['image2text']['model_id']

    logger.info(f"Prompts: {prompts}")
    logger.info(f"Class names: {class_names}")
    logger.info(f"Model ID: {model_id}")

    dataset_to_process = {
        'Train': (config['train_loader']['args']['root'], config['train_loader']['args']['detect_file']),
        'Validation': (config['val_loader']['args']['root'], config['val_loader']['args']['detect_file']),
        'Test': (config['test_loader']['args']['root'], config['test_loader']['args']['detect_file'])
    }

    for name, (root, filepath) in dataset_to_process.items():
        logger.info(f"===== Starting to process {name} set =====")
        run_inference_on_file(root, filepath, config, prompts, max_new_tokens, batch_size, device_ids, logger)
        logger.info(f"===== Finished processing {name} set =====")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Pytorch Template Preprocessing')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='Config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (e.g. 0,1,2), default: all available')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-b', '--batch_size', default=None, type=int,
                        help='batch size for inference (default: from config)')
    args.add_argument('-p', '--prompt_file', default=None, type=str,
                        help='path to a text file containing the prompt template (use {class_names} to inject classes)')

    config = ConfigParser.from_args(args)
    parsed_args = args.parse_args() 
    
    if parsed_args.device is not None:
        device_ids = [int(i) for i in parsed_args.device.split(',')]
    else:
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
        else:
            print("CUDA is not available. This script requires GPUs. Exiting.")
            exit()
            
    if not device_ids:
        print("No GPUs specified or detected. Exiting.")
        exit()

    if parsed_args.batch_size is not None:
        if 'image2text' not in config.config:
            config.config['image2text'] = {}
        config.config['image2text']['batch_size'] = parsed_args.batch_size
    
    main(config, device_ids, parsed_args.prompt_file)