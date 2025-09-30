import argparse
import torch
import os
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.reshape_transforms import swinT_reshape_transform
from tqdm import tqdm
import cv2
import random
import torch.nn.functional as F
from scipy import ndimage
import warnings


warnings.filterwarnings('ignore')


def swin_reshape_transform(tensor, height=7, width=7):
    # B, B_size, Chanel
    result = tensor 

    # reshape into b, c, h, w
    result = result.permute(0, 3, 1, 2)

    return result


def smooth_grad_cam(grayscale_cam, sigma=2.0):
    """
    Apply Gaussian smoothing to reduce grid artifacts
    """
    return ndimage.gaussian_filter(grayscale_cam, sigma=sigma)


def guided_backprop_hook(module, grad_input, grad_output):
    """
    Hook for guided backpropagation to get cleaner gradients
    """
    if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(grad_input[0], min=0.0),)


def main(config):
    logger = config.get_logger('detect')

    config['test_loader']['args']['batch_size'] = 1

    data_loader = config.init_obj('test_loader', module_data)
    model = config.init_obj('arch', module_arch)

    logger.info(model)
    logger.info('loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # --- Multi-branch CAM Configuration ---
    grad_cam_config = config['grad_cam']
    output_dir = grad_cam_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    cam_method = grad_cam_config.get('method', 'GradCAM')
    use_guided_backprop = grad_cam_config.get('guided_backprop', False)
    smooth_cam = grad_cam_config.get('smooth_cam', True)
    
    target_layers_config = grad_cam_config['target_layers']
    
    target_layers = {}
    for branch_name, layer_path in target_layers_config.items():
        try:
            target_layers[branch_name] = [eval(f"model.{layer_path}")]
            logger.info(f"Branch '{branch_name}' target layer: {layer_path}")
        except Exception as e:
            logger.error(f"Error parsing target layer for {branch_name}: {e}")
            continue
    
    if not target_layers:
        logger.error("No valid target layers found. Exiting.")
        return

    # Wrapper classes (giữ nguyên)
    class FaceModelWrapper(torch.nn.Module):
        def __init__(self, model, context_sample, body_sample):
            super().__init__()
            self.model = model
            self.context_sample = context_sample
            self.body_sample = body_sample
            
        def forward(self, face):
            return self.model(face, self.body_sample, self.context_sample)

    class BodyModelWrapper(torch.nn.Module):
        def __init__(self, model, face_sample, context_sample):
            super().__init__()
            self.model = model
            self.face_sample = face_sample
            self.context_sample = context_sample
            
        def forward(self, body):
            return self.model(self.face_sample, body, self.context_sample)

    class ContextModelWrapper(torch.nn.Module):
        def __init__(self, model, face_sample, body_sample):
            super().__init__()
            self.model = model
            self.face_sample = face_sample
            self.body_sample = body_sample
            
        def forward(self, context):
            return self.model(self.face_sample, self.body_sample, context)

    class_names = config['class_names']
    
    num_random_samples = grad_cam_config.get('num_samples', 10)
    
    total_samples = len(data_loader)
    if num_random_samples >= total_samples:
        selected_indices = list(range(total_samples))
    else:
        selected_indices = random.sample(range(total_samples), num_random_samples)
    
    logger.info(f"Selected {len(selected_indices)} random samples from {total_samples} total samples")
    logger.info(f"Using CAM method: {cam_method}")

    if use_guided_backprop:
        hooks = []
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                hooks.append(module.register_backward_hook(guided_backprop_hook))
    
    # --- Main processing loop ---
    for i, (data, target) in enumerate(tqdm(data_loader, desc="Processing samples")):
        if i not in selected_indices:
            continue
            
        face, body, context, target = data['face'].to(device), data['body'].to(device), data['context'].to(device), target.to(device)
        input_tensor_tuple = (face, body, context)
        output = model(*input_tensor_tuple)

        pred_index = torch.argmax(output, dim=1)[0].item()
        true_index = target[0].item()
        pred_class_name = class_names[pred_index]
        true_class_name = class_names[true_index]
        
        logger.info(f"Sample {i}: True={true_class_name}, Pred={pred_class_name}")
        
        branches_info = {
            'face': {'wrapper_class': FaceModelWrapper, 'wrapper_args': (model, context, body), 'input': face, 'tensor': face[0]},
            'body': {'wrapper_class': BodyModelWrapper, 'wrapper_args': (model, face, context), 'input': body, 'tensor': body[0]},
            'context': {'wrapper_class': ContextModelWrapper, 'wrapper_args': (model, face, body), 'input': context, 'tensor': context[0]}
        }
        
        for branch_name, branch_info in branches_info.items():
            if branch_name not in target_layers:
                logger.warning(f"Skipping branch '{branch_name}' - no target layer defined")
                continue
                
            try:
                wrapped_model = branch_info['wrapper_class'](*branch_info['wrapper_args'])
                wrapped_model.eval()
                
                branch_input = branch_info['input'].clone()
                branch_input.requires_grad_(True)
                
                with torch.no_grad():
                    test_output = wrapped_model(branch_input)
                    if test_output is None:
                        logger.warning(f"Model output is None for branch '{branch_name}', skipping...")
                        continue
                
                # <<< BẮT ĐẦU THAY ĐỔI 1 >>>
                # Thêm điều kiện để chọn reshape_transform
                # Chỉ áp dụng swin_reshape_transform cho nhánh 'body'
                reshape_for_branch = None
                if branch_name == 'body':
                    reshape_for_branch = swin_reshape_transform
                # <<< KẾT THÚC THAY ĐỔI 1 >>>

                try:
                    # <<< BẮT ĐẦU THAY ĐỔI 2 >>>
                    # Sử dụng biến `reshape_for_branch` đã được xác định ở trên
                    if cam_method == 'GradCAM':
                        cam = GradCAM(model=wrapped_model, 
                                      target_layers=target_layers[branch_name],
                                      reshape_transform=reshape_for_branch)
                    elif cam_method == 'GradCAMPlusPlus':
                        cam = GradCAMPlusPlus(model=wrapped_model, 
                                              target_layers=target_layers[branch_name],
                                              reshape_transform=reshape_for_branch)
                    else:
                        cam = GradCAM(model=wrapped_model, 
                                      target_layers=target_layers[branch_name],
                                      reshape_transform=reshape_for_branch)
                    # <<< KẾT THÚC THAY ĐỔI 2 >>>
                except Exception as cam_init_error:
                    logger.error(f"Failed to initialize CAM for branch '{branch_name}': {cam_init_error}")
                    continue
                
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                targets = [ClassifierOutputTarget(pred_index)]
                
                try:
                    grayscale_cam = cam(input_tensor=branch_input, 
                                        targets=targets,
                                        aug_smooth=False,
                                        eigen_smooth=False)
                except Exception as cam_error:
                    logger.warning(f"CAM failed for branch '{branch_name}', trying without targets...")
                    try:
                        grayscale_cam = cam(input_tensor=branch_input, 
                                            targets=None,
                                            aug_smooth=False,
                                            eigen_smooth=False)
                    except Exception as final_error:
                        logger.error(f"All CAM methods failed for branch '{branch_name}': {final_error}")
                        continue
                
                if grayscale_cam is None or len(grayscale_cam) == 0:
                    logger.warning(f"Empty CAM result for branch '{branch_name}', skipping...")
                    continue
                    
                grayscale_cam = grayscale_cam[0, :]
                
                if grayscale_cam is None or np.isnan(grayscale_cam).all():
                    logger.warning(f"Invalid CAM result for branch '{branch_name}', skipping...")
                    continue
                
                if smooth_cam and grayscale_cam is not None:
                    try:
                        grayscale_cam = smooth_grad_cam(grayscale_cam, sigma=1.5)
                    except Exception as smooth_error:
                        logger.warning(f"Smoothing failed for branch '{branch_name}': {smooth_error}")

                # --- Enhanced Visualization (giữ nguyên) ---
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                input_image_tensor = branch_info['tensor']
                if input_image_tensor.shape[0] == 1:
                    input_image_tensor = input_image_tensor.repeat(3, 1, 1)

                rgb_img = input_image_tensor.cpu().numpy().transpose((1, 2, 0))
                rgb_img = std * rgb_img + mean
                rgb_img = np.clip(rgb_img, 0, 1)
                
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)
                
                heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                original_img_bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


                filename_prefix = f"sample_{i}_true_{true_class_name}_pred_{pred_class_name}"
                
                original_path = os.path.join(output_dir, f"{filename_prefix}_original_{branch_name}.jpg")
                cam_path = os.path.join(output_dir, f"{filename_prefix}_cam_overlay_{branch_name}.jpg")
 
                
                cv2.imwrite(original_path, original_img_bgr)
                cv2.imwrite(cam_path, visualization_bgr)

                
                logger.info(f"   ✓ Saved {cam_method} for branch '{branch_name}'")
                
                del cam, wrapped_model, grayscale_cam, visualization, heatmap
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"   ✗ Error processing branch '{branch_name}' for sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if use_guided_backprop:
        for hook in hooks:
            hook.remove()

    logger.info(f"Multi-branch CAM generation finished. Results saved in '{output_dir}'.")
    logger.info(f"Generated {cam_method} visualizations for {len(target_layers)} branches: {list(target_layers.keys())}")
    logger.info(f"Processed {len(selected_indices)} samples")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Multi-branch CAM for Swin Transformer')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args)
    main(config)