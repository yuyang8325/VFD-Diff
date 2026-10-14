import os
import re
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CreateDatasetSynthesis(Dataset):
    """
    Synthesis dataset class: follows the naming convention
    tranA/phase/[a]_[b].png -> tranB/phase/[a]_mask_[b].png
    Arguments:
        phase: train/val/test -> the train/val/test subset of the dataset
        input_path: dataset root path (folder containing tranA/tranB)
        contrast1: source image directory name (e.g. tranA)
        contrast2: target image directory name (e.g. tranB)
    """
    def __init__(self, phase, input_path, contrast1, contrast2):
        # Base path configuration
        self.phase = phase  # train/val
        self.input_path = input_path
        self.src_root_dir = os.path.join(input_path, contrast1, phase)  # tranA/train or tranA/val
        self.tgt_root_dir = os.path.join(input_path, contrast2, phase)  # tranB/train or tranB/val
        
        # Verify that the base directories exist
        if not os.path.exists(self.src_root_dir):
            raise FileNotFoundError(f"Source image directory does not exist: {self.src_root_dir}")
        if not os.path.exists(self.tgt_root_dir):
            raise FileNotFoundError(f"Target image directory does not exist: {self.tgt_root_dir}")
        
        # Collect all source image paths (png/jpg/jpeg, case-insensitive)
        self.src_img_paths = self._get_all_img_paths(self.src_root_dir)
        if len(self.src_img_paths) == 0:
            raise ValueError(f"No image files found in the source directory {self.src_root_dir} (png/jpg/jpeg supported)")
        
        # Match the corresponding target image paths (core: a_b.png -> a_mask_b.png rule)
        self.tgt_img_paths = self._match_tgt_img_paths(self.src_img_paths, self.src_root_dir, self.tgt_root_dir)
        
        # Image preprocessing (matches the model input of train.py: single channel, normalized to [-1,1])
        self.transform = self._get_transform()

    def _get_all_img_paths(self, root_dir):
        """Collect the paths of all image files in a directory (png/jpg/jpeg, case-insensitive)"""
        img_extensions = ['png', 'jpg', 'jpeg']
        all_paths = []
        for ext in img_extensions:
            # Match both cases (e.g. PNG/Png/jpg/JPG)
            paths_lower = glob(os.path.join(root_dir, f'*.{ext.lower()}'))
            paths_upper = glob(os.path.join(root_dir, f'*.{ext.upper()}'))
            all_paths.extend(paths_lower + paths_upper)
        # Deduplicate and sort (to guarantee a consistent order)
        all_paths = sorted(list(set(all_paths)))
        return all_paths

    def _match_tgt_img_paths(self, src_img_paths, src_root_dir, tgt_root_dir):
        """
        Core matching logic:
        source file: xxx/[a]_[b].png -> target file: xxx/[a]_mask_[b].png
        """
        tgt_img_paths = []
        # Regex for "number_number.ext" (multi-digit numbers supported, e.g. 10_2.png -> 10_mask_2.png)
        pattern = re.compile(r'^(?P<a>\d+)_(?P<b>\d+)\.(?P<ext>png|jpg|jpeg)$', re.IGNORECASE)
        
        for src_path in src_img_paths:
            # Extract the source filename (e.g. 1_1.png)
            src_basename = os.path.basename(src_path)
            # Match the regex and split into a, b, extension
            match = pattern.match(src_basename)
            if not match:
                raise ValueError(
                    f"Source filename {src_basename} does not follow the rule: it must be in the form 'number_number.ext' (e.g. 1_1.png)"
                )
            a = match.group('a')
            b = match.group('b')
            ext = match.group('ext').lower()  # normalize the extension to lowercase (so PNG/png still match)
            
            # Build the target filename (a_mask_b.ext)
            tgt_basename = f"{a}_mask_{b}.{ext}"
            # Build the full target path
            tgt_path = os.path.join(tgt_root_dir, tgt_basename)
            
            # Verify that the target file exists
            if not os.path.exists(tgt_path):
                raise FileNotFoundError(f"Target image for source image {src_path} does not exist: {tgt_path}")
            
            tgt_img_paths.append(tgt_path)
        
        return tgt_img_paths

    def _get_transform(self):
        """Image preprocessing: single channel, resize, normalize to [-1,1] (GAN/diffusion model input)"""
        transform_list = [
            transforms.Resize((512, 512)),  # adjust together with --image_size in train.py; default 512
            transforms.Grayscale(num_output_channels=1),  # convert to single channel (fundus images)
            transforms.ToTensor(),  # convert to tensor [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # normalize to [-1,1]
        ]
        return transforms.Compose(transform_list)

    def __len__(self):
        """Dataset length: the number of source/target images (they are paired)"""
        return len(self.src_img_paths)

    def __getitem__(self, index):
        """
        Load a single sample: returns the source image (x1) and the target image (x2),
        both as single-channel tensors
        Return format: x1 (torch.Tensor) -> [1, H, W], x2 (torch.Tensor) -> [1, H, W]
        """
        # Load the source image
        src_path = self.src_img_paths[index]
        try:
            src_img = Image.open(src_path).convert('RGB')  # convert to RGB first, then to single channel (handles grayscale/color images)
        except Exception as e:
            raise IOError(f"Failed to load source image: {src_path}, error: {str(e)}")
        
        # Load the target image
        tgt_path = self.tgt_img_paths[index]
        try:
            tgt_img = Image.open(tgt_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load target image: {tgt_path}, error: {str(e)}")
        
        # Apply preprocessing
        x1 = self.transform(src_img)
        x2 = self.transform(tgt_img)
        
        return x1, x2

# Quick self-test (optional: verifies the pairing logic when running dataset.py directly)
if __name__ == "__main__":
    # Defaults to the current directory (tranA/tranB should sit in the repository root);
    # replace with your actual data path if needed
    test_input_path = "./"
    test_phase = "train"
    test_contrast1 = "tranA"
    test_contrast2 = "tranB"
    
    try:
        dataset = CreateDatasetSynthesis(
            phase=test_phase,
            input_path=test_input_path,
            contrast1=test_contrast1,
            contrast2=test_contrast2
        )
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
        # Test loading the first sample
        x1, x2 = dataset[0]
        print(f"Source image tensor shape: {x1.shape}, target image tensor shape: {x2.shape}")
        print(f"Source image value range: [{x1.min():.4f}, {x1.max():.4f}]")
        print(f"Target image value range: [{x2.min():.4f}, {x2.max():.4f}]")
    except Exception as e:
        print(f"Failed to load the dataset: {str(e)}")
