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

def read_image(image_path):
    """
    the functions reads an image from the given path and returns a PIL Image object
    """
    return Image.open(image_path).convert('RGB')

def get_assistant_text(text):
    start_index = text.find("ASSISTANT")
    assistant_text = text[start_index:] if start_index != -1 else ""
    return assistant_text

def prepare_batch_data(lines: List[str], root: str, processed_images: set, batch_size: int, logger):
    """
    Prepere batch data for processing
    """
    batch_data = []
    current_batch = []
    
    for line in lines:
        try:
            parts = line.strip().split(',')
            if not parts or not parts[0]:
                continue
            
            image_path = parts[0]
            
            # Skip already processed images
            if image_path in processed_images:
                continue
            
            face_coords = tuple(map(int, parts[2:6]))
            full_image_path = os.path.join(root, image_path)
            
            if not os.path.exists(full_image_path):
                logger.warning(f"Image {full_image_path} does not exist. Skipping.")
                continue
            
            current_batch.append({
                'original_line': line.strip(),
                'image_path': image_path,
                'full_image_path': full_image_path,
                'face_coords': face_coords
            })
            
            # When reaching batch_size, add to bach_dât
            if len(current_batch) == batch_size:
                batch_data.append(current_batch)
                current_batch = []
                
        except Exception as e:
            logger.error(f"Failed to parse line: {line.strip()}. Error: {e}")
            continue
    
    # Add the last batch if it has remaining items
    if current_batch:
        batch_data.append(current_batch)
    
    return batch_data

def process_image_batch(batch_items: List[dict], pipe, prompts: str, max_new_tokens: int, logger):
    """
    Process a batch of images together
    """
    try:
        # Prepare images for the batch
        images = []
        for item in batch_items:
            # Read image and draw box
            image = read_image(item['full_image_path'])
            x1, y1, x2, y2 = item['face_coords']
            draw = ImageDraw.Draw(image)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            images.append(image)
        
        # Create a list of prompt for each image in the batch
        batch_prompts = [prompts] * len(images)

        # Call the pipeline with the batch
        results = pipe(
            images, 
            text=batch_prompts, 
            generate_kwargs={"max_new_tokens": max_new_tokens},
            batch_size=len(images)
        )
        
        # Process results
        output_lines = []
        for i, result in enumerate(results):
            output_prompt = result["generated_text"]
            output_prompt = get_assistant_text(output_prompt)
            output_line = f"{batch_items[i]['original_line']},{output_prompt}"
            output_lines.append(output_line)
        
        return output_lines
        
    except Exception as e:
        logger.error(f"Failed to process batch. Error: {e}")
        # Fallback: Process images one by one
        output_lines = []
        for item in batch_items:
            try:
                image = read_image(item['full_image_path'])
                x1, y1, x2, y2 = item['face_coords']
                draw = ImageDraw.Draw(image)
                draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
                
                result = pipe(image, text=prompts, generate_kwargs={"max_new_tokens": max_new_tokens})
                output_prompt = result[0]["generated_text"]
                output_prompt = get_assistant_text(output_prompt)
                output_line = f"{item['original_line']},{output_prompt}"
                output_lines.append(output_line)
                
            except Exception as e_single:
                logger.error(f"Failed to process single image {item['image_path']}. Error: {e_single}")
                continue
        
        return output_lines

### THAY ĐỔI ###: Sửa hàm worker để nhận Lock và ghi trực tiếp vào file output
def process_worker(gpu_id, batch_chunks, model_id, prompts, max_new_tokens, output_file, lock, config):
    """
    Hàm worker, mỗi worker sẽ chạy trên một tiến trình và một GPU riêng biệt.
    Sử dụng Lock để ghi trực tiếp vào file output chung.
    """
    logger = config.get_logger(f'worker_gpu_{gpu_id}')
    device = torch.device(f'cuda:{gpu_id}')
    logger.info(f"Worker on GPU {gpu_id} started, processing {len(batch_chunks)} batches.")

    model_kwargs = {
        "torch_dtype": torch.float16,
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

    progress_bar = tqdm(batch_chunks, desc=f"GPU {gpu_id}", position=gpu_id)
    for batch_items in progress_bar:
        output_lines = process_image_batch(batch_items, pipe, prompts, max_new_tokens, logger)
        
        # Sử dụng lock để đảm bảo chỉ 1 tiến trình được ghi file tại 1 thời điểm
        with lock:
            # Mở file output chung ở chế độ 'a' (append - ghi nối)
            with open(output_file, 'a', encoding='utf-8') as f_out:
                for output_line in output_lines:
                    f_out.write(f"{output_line}\n")
    
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

    logger.info("Preparing batch data...")
    batch_data = prepare_batch_data(all_lines, root, processed_images, batch_size, logger)
    logger.info(f"Total batches to process: {len(batch_data)}")
    
    if not batch_data:
        logger.info("No new data to process.")
        return

    num_gpus = len(device_ids)
    chunks = np.array_split(batch_data, num_gpus) 
    
    # Tạo Lock để chia sẻ giữa các tiến trình
    manager = mp.Manager()
    lock = manager.Lock()
    
    worker_args = []
    for i, gpu_id in enumerate(device_ids):
        chunk_list = [list(batch) for batch in chunks[i]]
        if not chunk_list: continue
        
        worker_args.append((
            gpu_id, 
            chunk_list,
            config['image2text']['model_id'],
            prompts,
            max_new_tokens,
            output_file,  # Truyền đường dẫn file output cuối cùng
            lock,         # Truyền đối tượng lock
            config
        ))

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        pool.starmap(process_worker, worker_args)

    # Không cần gộp file nữa
    logger.info(f"All workers finished. Results have been saved to {output_file}.")


def main(config, device_ids):
    logger = config.get_logger('image2text')
    logger.info("Starting inference process...")
    logger.info(f"Using devices: {device_ids}")

    class_names = config['class_names']
    max_new_tokens = config['image2text']['max_new_tokens']
    
    batch_size = config['image2text'].get('batch_size', 8)
    logger.info(f"Using batch size per GPU: {batch_size}")
    
    prompts = f"USER: <image>\nGiven the following list of emotions:{', '.join(class_names)}. Based on the image context, please choose which emotions are more suitable for describing how the person in the red box feels and explain in detail why you choose these emotions according to the aspects: actions and postures of the person in the red box, the context surrounding the person in the red box.\nASSISTANT:"
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
    
    main(config, device_ids)