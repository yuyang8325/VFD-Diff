import os
import re
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CreateDatasetSynthesis(Dataset):
    """
    合成数据集类：适配tranA/phase/[a]_[b].png → tranB/phase/[a]_mask_[b].png的命名规则
    输入参数：
        phase: train/val/test → 对应数据集的训练/验证/测试子集
        input_path: 数据集根路径（包含tranA/tranB的文件夹）
        contrast1: 源图像目录名（如tranA）
        contrast2: 目标图像目录名（如tranB）
    """
    def __init__(self, phase, input_path, contrast1, contrast2):
        # 基础路径配置
        self.phase = phase  # train/val
        self.input_path = input_path
        self.src_root_dir = os.path.join(input_path, contrast1, phase)  # tranA/train or tranA/val
        self.tgt_root_dir = os.path.join(input_path, contrast2, phase)  # tranB/train or tranB/val
        
        # 校验基础目录是否存在
        if not os.path.exists(self.src_root_dir):
            raise FileNotFoundError(f"源图像目录不存在 → {self.src_root_dir}")
        if not os.path.exists(self.tgt_root_dir):
            raise FileNotFoundError(f"目标图像目录不存在 → {self.tgt_root_dir}")
        
        # 获取所有源图像路径（支持png/jpg/jpeg，忽略大小写）
        self.src_img_paths = self._get_all_img_paths(self.src_root_dir)
        if len(self.src_img_paths) == 0:
            raise ValueError(f"源图像目录{self.src_root_dir}下未找到任何图像文件（支持png/jpg/jpeg）")
        
        # 匹配对应的目标图像路径（核心：适配a_b.png → a_mask_b.png规则）
        self.tgt_img_paths = self._match_tgt_img_paths(self.src_img_paths, self.src_root_dir, self.tgt_root_dir)
        
        # 图像预处理（适配train.py的模型输入，单通道、归一化到[-1,1]）
        self.transform = self._get_transform()

    def _get_all_img_paths(self, root_dir):
        """获取目录下所有图像文件路径（支持png/jpg/jpeg，忽略大小写）"""
        img_extensions = ['png', 'jpg', 'jpeg']
        all_paths = []
        for ext in img_extensions:
            # 匹配大小写（如PNG/Png/jpg/JPG）
            paths_lower = glob(os.path.join(root_dir, f'*.{ext.lower()}'))
            paths_upper = glob(os.path.join(root_dir, f'*.{ext.upper()}'))
            all_paths.extend(paths_lower + paths_upper)
        # 去重并排序（保证顺序一致）
        all_paths = sorted(list(set(all_paths)))
        return all_paths

    def _match_tgt_img_paths(self, src_img_paths, src_root_dir, tgt_root_dir):
        """
        核心匹配逻辑：
        源文件：xxx/[a]_[b].png → 目标文件：xxx/[a]_mask_[b].png
        """
        tgt_img_paths = []
        # 正则匹配「数字_数字.后缀」（支持多位数，如10_2.png → 10_mask_2.png）
        pattern = re.compile(r'^(?P<a>\d+)_(?P<b>\d+)\.(?P<ext>png|jpg|jpeg)$', re.IGNORECASE)
        
        for src_path in src_img_paths:
            # 提取源文件名（如1_1.png）
            src_basename = os.path.basename(src_path)
            # 匹配正则，拆分a、b、后缀
            match = pattern.match(src_basename)
            if not match:
                raise ValueError(
                    f"源文件名{src_basename}不符合规则 → 需为「数字_数字.后缀」格式（如1_1.png）"
                )
            a = match.group('a')
            b = match.group('b')
            ext = match.group('ext').lower()  # 统一后缀为小写（避免PNG/png不匹配）
            
            # 拼接目标文件名（a_mask_b.ext）
            tgt_basename = f"{a}_mask_{b}.{ext}"
            # 拼接目标文件完整路径
            tgt_path = os.path.join(tgt_root_dir, tgt_basename)
            
            # 校验目标文件是否存在
            if not os.path.exists(tgt_path):
                raise FileNotFoundError(f"源图像{src_path}对应的目标图像不存在 → {tgt_path}")
            
            tgt_img_paths.append(tgt_path)
        
        return tgt_img_paths

    def _get_transform(self):
        """图像预处理：单通道、resize、归一化到[-1,1]（适配GAN/扩散模型输入）"""
        transform_list = [
            transforms.Resize((512, 512)),  # 可根据train.py的--image_size调整，默认512
            transforms.Grayscale(num_output_channels=1),  # 转为单通道（眼底图像）
            transforms.ToTensor(),  # 转为张量 [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
        ]
        return transforms.Compose(transform_list)

    def __len__(self):
        """数据集长度：源/目标图像数量一致"""
        return len(self.src_img_paths)

    def __getitem__(self, index):
        """
        加载单样本：返回源图像(x1)和目标图像(x2)，均为单通道张量
        返回格式：x1 (torch.Tensor) → [1, H, W], x2 (torch.Tensor) → [1, H, W]
        """
        # 加载源图像
        src_path = self.src_img_paths[index]
        try:
            src_img = Image.open(src_path).convert('RGB')  # 先转RGB再转单通道（兼容灰度/彩色图）
        except Exception as e:
            raise IOError(f"加载源图像失败 → {src_path}, 错误：{str(e)}")
        
        # 加载目标图像
        tgt_path = self.tgt_img_paths[index]
        try:
            tgt_img = Image.open(tgt_path).convert('RGB')
        except Exception as e:
            raise IOError(f"加载目标图像失败 → {tgt_path}, 错误：{str(e)}")
        
        # 应用预处理
        x1 = self.transform(src_img)
        x2 = self.transform(tgt_img)
        
        return x1, x2

# 测试代码（可选：运行dataset.py时验证配对逻辑）
if __name__ == "__main__":
    # 默认使用当前目录（tranA/tranB应位于本仓库根目录），也可替换为实际数据路径
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
        print(f"数据集加载成功！总样本数：{len(dataset)}")
        # 测试加载第一个样本
        x1, x2 = dataset[0]
        print(f"源图像张量形状：{x1.shape}，目标图像张量形状：{x2.shape}")
        print(f"源图像值范围：[{x1.min():.4f}, {x1.max():.4f}]")
        print(f"目标图像值范围：[{x2.min():.4f}, {x2.max():.4f}]")
    except Exception as e:
        print(f"数据集加载失败：{str(e)}")