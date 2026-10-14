import argparse
import torch
import numpy as np
import os
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr

# VFD-Diff core modules (paper Eqs. 1-9)
from backbones.vfd_modules import (
    adaptive_frangi_mask, hf_enhanced_noise_like, fourier_low_freq_suppress,
    masked_l1_loss, masked_ssim_loss, feature_distill_loss, mask_weighted_source,
)

# Utility for copying files
def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

#%% Custom dataset class (modified: also returns image filenames + VFD-Diff vessel masks)
class CustomFFADataset(Dataset):
    def __init__(self, root_path, phase="train", contrast1="tranA", contrast2="tranB", transform=None,
                 compute_mask=True, frangi_dmax=16.0, frangi_dmin=2.0):
        """
        Load the tranA/tranB dataset (auto-discovers class folders, follows the
        X_Y.png <-> X_mask_Y.png naming convention)
        :param root_path: root path (folder containing tranA/tranB)
        :param phase: train/val
        :param contrast1: source modality folder name (tranA: color fundus photography)
        :param contrast2: target modality folder name (tranB: fluorescein angiography)
        :param transform: image transforms
        :param compute_mask: whether to compute the VFD-Diff vessel mask M_VSA (paper Eqs. 1-4, with disk cache)
        :param frangi_dmax/frangi_dmin: max/min vessel diameter in Eqs. (2)(4) (pixels, 512-based)
        """
        self.root_path = root_path
        self.phase = phase
        self.contrast1 = contrast1
        self.contrast2 = contrast2
        self.transform = transform if transform is not None else self._default_transform()
        self.compute_mask = compute_mask
        self.frangi_dmax = frangi_dmax
        self.frangi_dmin = frangi_dmin
        # Vessel-mask cache dir: <root>/<contrast1>_vmasks/<phase>/... (same name as the image, .png)
        self.mask_root = os.path.join(root_path, contrast1 + '_vmasks', phase)

        # Collect all matched image pairs (now including filenames)
        self.image_pairs = self._collect_image_pairs()

    def _default_transform(self):
        """Default transform: tailored for 512x512 single-channel fundus images"""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
        ])

    def _get_tranB_filename(self, tranA_filename):
        """
        Core helper: convert a tranA filename to its tranB counterpart
        Rule: tranA `1_1.png` -> tranB `1_mask_1.png`
              tranA `5_10.png` -> tranB `5_mask_10.png`
        """
        name, ext = os.path.splitext(tranA_filename)
        name_parts = name.split('_')
        
        if len(name_parts) >= 2:
            tranB_name_parts = name_parts[:1] + ['mask'] + name_parts[1:]
            tranB_name = '_'.join(tranB_name_parts)
        else:
            tranB_name = name
            print(f"Warning: tranA filename {tranA_filename} does not follow the 'prefix_index' convention")
        
        return tranB_name + ext

    def _collect_image_pairs(self):
        """
        Traverse folders to collect image pairs:
        - train: auto-discover class subfolders + name-convention matching
        - val: directly in the root folder + name-convention matching
        Returns: a list of tuples (c1_path, c2_path, c1_filename)
        """
        image_pairs = []
        contrast1_root = os.path.join(self.root_path, self.contrast1, self.phase)
        contrast2_root = os.path.join(self.root_path, self.contrast2, self.phase)
        
        if not os.path.exists(contrast1_root) or not os.path.exists(contrast2_root):
            raise ValueError(f"Path does not exist: {contrast1_root} or {contrast2_root}")
        
        if self.phase == "train":
            class_folders = [d for d in os.listdir(contrast1_root) 
                             if os.path.isdir(os.path.join(contrast1_root, d))]
            
            if not class_folders:
                print(f"Warning: no class subfolders found under {contrast1_root}")
            else:
                print(f"Discovered class folders: {class_folders}")
            
            for class_folder in class_folders:
                c1_class_path = os.path.join(contrast1_root, class_folder)
                c2_class_path = os.path.join(contrast2_root, class_folder)
                
                if not os.path.exists(c2_class_path):
                    print(f"Warning: class folder {class_folder} missing under tranB, skipped")
                    continue
                
                c1_images = [f for f in os.listdir(c1_class_path) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_name in c1_images:
                    c1_img_path = os.path.join(c1_class_path, img_name)
                    tranB_img_name = self._get_tranB_filename(img_name)
                    c2_img_path = os.path.join(c2_class_path, tranB_img_name)
                    
                    if os.path.exists(c2_img_path):
                        # Modified: also store the filename (without path)
                        image_pairs.append((c1_img_path, c2_img_path, img_name))
                    else:
                        print(f"Warning: file mismatch\n  tranA: {c1_img_path}\n  expected tranB: {c2_img_path}")
        
        elif self.phase == "val":
            c1_images = [f for f in os.listdir(contrast1_root) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in c1_images:
                c1_img_path = os.path.join(contrast1_root, img_name)
                tranB_img_name = self._get_tranB_filename(img_name)
                c2_img_path = os.path.join(contrast2_root, tranB_img_name)
                
                if os.path.exists(c2_img_path):
                    image_pairs.append((c1_img_path, c2_img_path, img_name))
                else:
                    print(f"Warning: val-set file mismatch:\n  tranA: {c1_img_path}\n  expected tranB: {c2_img_path}")
        
        if len(image_pairs) == 0:
            raise ValueError(
                f"\n{'='*60}\n"
                f"Error: no matched image pairs found for the {self.phase} set!\n"
                f"{'='*60}\n"
                f"Please check the following:\n"
                f"1. Paths are correct:\n   tranA: {contrast1_root}\n   tranB: {contrast2_root}\n"
                f"2. Naming rule: tranA 'X_Y.png' must correspond to tranB 'X_mask_Y.png'\n"
                f"3. Folder structure: tranA and tranB must share the same class subfolders (train set)\n"
                f"{'='*60}"
            )
        
        print(f"\nSuccessfully loaded the {self.phase} set: {len(image_pairs)} image pairs\n")
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def _load_or_compute_mask(self, c1_path):
        """
        VFD-Diff vessel mask M_VSA (paper Eqs. 1-4):
        read from the disk cache if available; otherwise compute it with the
        parameter-adaptive Frangi filter and cache it as a PNG
        Returns: [1,H,W] float tensor (0/1)
        """
        contrast_root = os.path.join(self.root_path, self.contrast1, self.phase)
        rel = os.path.relpath(c1_path, contrast_root)
        mask_path = os.path.join(self.mask_root, os.path.splitext(rel)[0] + '.png')

        if os.path.exists(mask_path):
            m = Image.open(mask_path).convert('L').resize((512, 512))
            mask = (torch.from_numpy(np.array(m, dtype=np.float32) / 255.0) > 0.5).float()
            return mask.unsqueeze(0)

        img = Image.open(c1_path).convert('L').resize((512, 512))
        arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)[None, None]
        mask = adaptive_frangi_mask(arr, invert=False,
                                     d_max=self.frangi_dmax, d_min=self.frangi_dmin)
        try:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            Image.fromarray((mask[0, 0].numpy() * 255.0).astype(np.uint8)).save(mask_path)
        except OSError as e:
            print(f'Warning: failed to write the vessel-mask cache ({e}); computing directly each time instead')
        return mask[0]  # [1,H,W]

    def __getitem__(self, idx):
        # Modified: now returns four values: img1, img2, filename, vessel_mask
        c1_path, c2_path, filename = self.image_pairs[idx]

        try:
            img1 = Image.open(c1_path).convert('L')
            img2 = Image.open(c2_path).convert('L')
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {c1_path} or {c2_path}, error: {e}")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        mask = self._load_or_compute_mask(c1_path) if self.compute_mask else torch.zeros_like(img1)
        return img1, img2, filename, mask

#%% Helper: overlay the filename onto an image
def add_filename_to_image(image_tensor, filename, font_size=20):
    """
    Add the filename above the image
    :param image_tensor: input image tensor [C, H, W] or [B, C, H, W]
    :param filename: filename to add
    :param font_size: font size
    :return: image tensor with the text drawn on it
    """
    # Handle the batched case
    if len(image_tensor.shape) == 4:
        result = []
        for i in range(image_tensor.shape[0]):
            result.append(add_filename_to_image(image_tensor[i], filename[i] if isinstance(filename, list) else filename, font_size))
        return torch.stack(result)
    
    # Single-image processing
    # Convert the tensor to a PIL image
    img_np = image_tensor.cpu().numpy()
    img_np = (img_np + 1) / 2.0  # convert from [-1,1] to [0,1]
    img_np = np.transpose(img_np, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    
    if img_np.shape[2] == 1:
        img_np = np.squeeze(img_np, axis=2)  # single channel
    
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    
    # If single-channel, convert to RGB so text can be drawn
    if img_pil.mode == 'L':
        img_pil = img_pil.convert('RGB')
    
    # Create a drawing object
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a font; fall back to the default font on failure
    try:
        # Try different font paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Arial.ttf"
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Compute the text position (centered, at the top)
    text_bbox = draw.textbbox((0, 0), filename, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Add a white bar above the image
    new_height = img_pil.height + text_height + 10
    new_img = Image.new('RGB', (img_pil.width, new_height), color='white')
    new_img.paste(img_pil, (0, text_height + 10))
    
    # Draw the text on the white bar
    draw = ImageDraw.Draw(new_img)
    text_x = (img_pil.width - text_width) // 2
    text_y = 5
    draw.text((text_x, text_y), filename, fill='black', font=font)
    
    # Convert back to a tensor
    new_img_np = np.array(new_img).astype(np.float32) / 255.0
    new_img_np = np.transpose(new_img_np, (2, 0, 1))  # [H, W, C] -> [C, H, W]
    
    # If the input is single-channel, keep the output single-channel
    if image_tensor.shape[0] == 1:
        new_img_np = new_img_np[[0], :, :]  # keep only the first channel
    
    return torch.from_numpy(new_img_np) * 2.0 - 1.0  # convert back to [-1,1]

#%% Diffusion-coefficient helpers (unchanged)
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t).to(device) * (1. - eps_small) + eps_small
    return t

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
    
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t).to(device) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8, device=device)
    betas = betas.to(device)
    
    betas = torch.cat((first[None], betas))
    
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = torch.cumprod(self.a_s, dim=0)
        self.sigmas_cum = torch.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

def q_sample(coeff, x_start, t, *, noise=None):
    if noise is None:
      noise = torch.randn_like(x_start)
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    return x_t

def q_sample_pairs(coeff, x_start, t, noise_fn=None):
    """
    Forward noising to obtain sample pairs at two adjacent timesteps.
    When noise_fn is not None, VFD-Diff FDE is enabled (paper Eq. 7):
    the standard Gaussian noise n is replaced by high-frequency-enhanced noise.
    """
    if noise_fn is not None:
        noise = noise_fn(x_start)
        noise_tp1 = noise_fn(x_start)
    else:
        noise = torch.randn_like(x_start)
        noise_tp1 = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t, noise=noise)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise_tp1
    return x_t, x_t_plus_one

#%% Posterior sampling (unchanged)
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
    
    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt, post_fn=None):
    """Reverse-diffusion sampling. When post_fn is not None, the final output is
    post-processed on the generation side (VFD-Diff FDE Eq. 8: Fourier low-frequency suppression)"""
    x = x_init[:,[0],:]
    source = x_init[:,[1],:]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,[0],:], x, t)
            x = x_new.detach()
    if post_fn is not None:
        x = post_fn(x)
    return x

#%% LR schedule: baseline cosine / paper Sec. 3.3 (constant first half + linear decay to 0 in the second half)
def make_lr_scheduler(optimizer, args):
    if args.lr_schedule == 'paper':
        half = max(1, args.num_epoch // 2)
        tail = max(1, args.num_epoch - half)
        def lr_lambda(ep):
            return 1.0 if ep < half else max(0.0, 1.0 - (ep - half) / tail)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

#%% VFD-Diff teacher-stage training (paper Sec. 2.3)
def train_teacher_stage(args, device, data_loader_train, data_loader_val,
                        use_vmg, noise_fn, lf_suppress, to_range_0_1):
    """
    The teacher network shares the architecture of the student gen_diffusive_2
    (NCSNpp, generates tranB) and differs only in its input: the condition image
    is the vessel-mask-weighted image (paper: the teacher takes vessel-mask-weighted
    local regions). It independently trains a mirrored baseline CFP->FFA pipeline
    (diffusion generator + bidirectional direct generators + 2 discriminators),
    and saves teacher_vfd.pth at the end for frozen distillation in the student stage.
    """
    from backbones.discriminator import Discriminator_large
    from backbones.ncsnpp_generator_adagn import NCSNpp
    import backbones.generator_resnet
    from utils.EMA import EMA

    nz = args.nz
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    # Teacher input weighting: DG requires the vessel-mask region to be fed to the
    # teacher network; weight it whenever the mask is available
    weight_input = use_vmg or (not args.no_dg)

    exp_path = os.path.join(args.output_path, args.exp)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    print("[VFD-Diff teacher stage] Initializing diffusion coefficients...")
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    # Build the networks (mirrored baseline tranA->tranB pipeline)
    t_gen_diff = NCSNpp(args).to(device)                 # teacher: diffusion generator (generates tranB)
    args.num_channels = 1
    t_gen_direct_f = backbones.generator_resnet.define_G(
        netG='resnet_6blocks', gpu_ids=[args.gpu_chose])  # tranA->tranB direct generator
    t_gen_direct_r = backbones.generator_resnet.define_G(
        netG='resnet_6blocks', gpu_ids=[args.gpu_chose])  # tranB->tranA direct generator (for the cycle)
    t_disc_diff = Discriminator_large(
        nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    t_disc_cycle = backbones.generator_resnet.define_D(gpu_ids=[args.gpu_chose])

    opt_t_diff = optim.Adam(t_gen_diff.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_t_df = optim.Adam(t_gen_direct_f.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_t_dr = optim.Adam(t_gen_direct_r.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_t_dd = optim.Adam(t_disc_diff.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_t_dc = optim.Adam(t_disc_cycle.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    if args.use_ema:
        opt_t_diff = EMA(opt_t_diff, ema_decay=args.ema_decay)
        opt_t_df = EMA(opt_t_df, ema_decay=args.ema_decay)
        opt_t_dr = EMA(opt_t_dr, ema_decay=args.ema_decay)
    sch_t_diff = make_lr_scheduler(opt_t_diff, args)
    sch_t_df = make_lr_scheduler(opt_t_df, args)
    sch_t_dr = make_lr_scheduler(opt_t_dr, args)
    sch_t_dd = make_lr_scheduler(opt_t_dd, args)
    sch_t_dc = make_lr_scheduler(opt_t_dc, args)

    global_step, epoch, init_epoch = 0, 0, 0
    teacher_content_file = os.path.join(exp_path, 'content_teacher.pth')
    if args.resume and os.path.exists(teacher_content_file):
        ckpt = torch.load(teacher_content_file, map_location=device)
        init_epoch = ckpt['epoch']
        epoch = init_epoch
        t_gen_diff.load_state_dict(ckpt['t_gen_diff'])
        t_gen_direct_f.load_state_dict(ckpt['t_gen_direct_f'])
        t_gen_direct_r.load_state_dict(ckpt['t_gen_direct_r'])
        t_disc_diff.load_state_dict(ckpt['t_disc_diff'])
        t_disc_cycle.load_state_dict(ckpt['t_disc_cycle'])
        opt_t_diff.load_state_dict(ckpt['opt_t_diff'])
        opt_t_df.load_state_dict(ckpt['opt_t_df'])
        opt_t_dr.load_state_dict(ckpt['opt_t_dr'])
        opt_t_dd.load_state_dict(ckpt['opt_t_dd'])
        opt_t_dc.load_state_dict(ckpt['opt_t_dc'])
        sch_t_diff.load_state_dict(ckpt['sch_t_diff'])
        sch_t_df.load_state_dict(ckpt['sch_t_df'])
        sch_t_dr.load_state_dict(ckpt['sch_t_dr'])
        sch_t_dd.load_state_dict(ckpt['sch_t_dd'])
        sch_t_dc.load_state_dict(ckpt['sch_t_dc'])
        global_step = ckpt['global_step']
        print(f"=> [teacher stage] loaded checkpoint (epoch {ckpt['epoch']})")

    for epoch in range(init_epoch, args.num_epoch + 1):
        t_gen_diff.train(); t_gen_direct_f.train(); t_gen_direct_r.train()
        t_disc_diff.train(); t_disc_cycle.train()

        for iteration, (x1, x2, filenames, vessel_mask) in enumerate(data_loader_train):
            real1 = x1.to(device, non_blocking=True)   # tranA (CFP)
            real2 = x2.to(device, non_blocking=True)   # tranB (FFA)
            vessel_mask = vessel_mask.to(device, non_blocking=True)
            # Teacher input: vessel-mask-weighted CFP (paper: vessel-mask-weighted local regions)
            if weight_input:
                w1 = mask_weighted_source(real1, vessel_mask, args.vfd_teacher_gamma)
            else:
                w1 = real1

            # ================= Discriminator step =================
            for p in t_disc_diff.parameters():
                p.requires_grad = True
            for p in t_disc_cycle.parameters():
                p.requires_grad = True
            if iteration % accumulation_steps == 0:
                t_disc_diff.zero_grad()
                t_disc_cycle.zero_grad()

            t2 = torch.randint(0, args.num_timesteps, (real2.size(0),), device=device)
            # FDE (Eq. 7): the teacher stage also uses high-frequency-enhanced noise
            # (forward process consistent with the student)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real2, t2, noise_fn=noise_fn)
            x2_t.requires_grad = True

            z2 = torch.randn(batch_size, nz, device=device)
            cond_r = t_gen_direct_r(real2)  # CFP predicted from real FFA used as the condition (mirrored baseline)
            if weight_input:
                cond_r = mask_weighted_source(cond_r, vessel_mask, args.vfd_teacher_gamma)
            t_x2_0 = t_gen_diff(torch.cat((x2_tp1.detach(), cond_r), axis=1), t2, z2)
            x2_pos = sample_posterior(pos_coeff, t_x2_0[:, [0], :], x2_tp1, t2)

            D_real = t_disc_diff(x2_t, t2, x2_tp1.detach()).view(-1)
            D_fake = t_disc_diff(x2_pos, t2, x2_tp1.detach()).view(-1)
            errD_real = F.softplus(-D_real).mean() / accumulation_steps
            errD_fake = F.softplus(D_fake).mean() / accumulation_steps

            grad_penalty = 0.0
            if args.lazy_reg is None or global_step % args.lazy_reg == 0:
                grad = torch.autograd.grad(outputs=D_real.sum(), inputs=x2_t, create_graph=True)[0]
                gp = (grad.view(grad.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = (args.r1_gamma / 2 * gp) / accumulation_steps

            fake2_direct = t_gen_direct_f(w1)
            Dc_real = t_disc_cycle(real2).view(-1)
            Dc_fake = t_disc_cycle(fake2_direct).view(-1)
            errD_cycle = (F.softplus(-Dc_real).mean() + F.softplus(Dc_fake).mean()) / accumulation_steps

            (errD_real + errD_fake + grad_penalty + errD_cycle).backward()

            if (iteration + 1) % accumulation_steps == 0:
                opt_t_dd.step()
                opt_t_dc.step()
                t_disc_diff.zero_grad()
                t_disc_cycle.zero_grad()

            if iteration % 10 == 0:
                torch.cuda.empty_cache()

            for p in t_disc_diff.parameters():
                p.requires_grad = False
            for p in t_disc_cycle.parameters():
                p.requires_grad = False

            # ================= Generator step =================
            if iteration % accumulation_steps == 0:
                t_gen_diff.zero_grad()
                t_gen_direct_f.zero_grad()
                t_gen_direct_r.zero_grad()

            t2 = torch.randint(0, args.num_timesteps, (real2.size(0),), device=device)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real2, t2, noise_fn=noise_fn)
            z2 = torch.randn(batch_size, nz, device=device)

            x1_fake = t_gen_direct_r(real2)
            x2_fake = t_gen_direct_f(w1)
            x2_rec = t_gen_direct_r(x2_fake)  # cycle: weighted CFP -> fake FFA -> reconstructed CFP

            cond = x1_fake
            if weight_input:
                cond = mask_weighted_source(x1_fake, vessel_mask, args.vfd_teacher_gamma)
            t_x2_0 = t_gen_diff(torch.cat((x2_tp1.detach(), cond), axis=1), t2, z2)
            x2_pos = sample_posterior(pos_coeff, t_x2_0[:, [0], :], x2_tp1, t2)

            errG_adv = F.softplus(-t_disc_diff(x2_pos, t2, x2_tp1.detach()).view(-1).mean()) / accumulation_steps
            errG_cycle_adv = F.softplus(-t_disc_cycle(x2_fake).view(-1).mean()) / accumulation_steps
            errG_cycle = F.l1_loss(x2_rec, real1) / accumulation_steps
            # VMG (Eqs. 5/6): vessel-mask-weighted L1 and SSIM
            if use_vmg:
                errG_l1 = masked_l1_loss(t_x2_0[:, [0], :], real2, vessel_mask, args.vfd_lam_l1) / accumulation_steps
                errG_ssim = masked_ssim_loss(t_x2_0[:, [0], :], real2, vessel_mask, args.vfd_lam_ssim) / accumulation_steps
            else:
                errG_l1 = F.l1_loss(t_x2_0[:, [0], :], real2) / accumulation_steps
                errG_ssim = torch.tensor(0.0, device=device)

            errG = args.lambda_l1_loss * errG_cycle + errG_adv + errG_cycle_adv \
                 + args.lambda_l1_loss * errG_l1 + args.vfd_ssim_weight * errG_ssim
            errG.backward()

            if (iteration + 1) % accumulation_steps == 0:
                opt_t_diff.step()
                opt_t_df.step()
                opt_t_dr.step()
                for p in t_disc_diff.parameters():
                    p.requires_grad = True
                for p in t_disc_cycle.parameters():
                    p.requires_grad = True

            global_step += 1

            if iteration % 100 == 0:
                print(
                    f'[teacher] epoch {epoch} iteration {iteration}, '
                    f'G-Cycle: {errG_cycle.item() * accumulation_steps:.4f}, '
                    f'G-L1: {errG_l1.item() * accumulation_steps:.4f}, '
                    f'G-SSIM: {errG_ssim.item() * accumulation_steps:.4f}, '
                    f'G-Sum: {errG.item() * accumulation_steps:.4f}, '
                    f'D Loss: {(errD_real + errD_fake).item() * accumulation_steps:.4f}'
                )

        if not args.no_lr_decay:
            sch_t_diff.step(); sch_t_df.step(); sch_t_dr.step()
            sch_t_dd.step(); sch_t_dc.step()

        if epoch % 10 == 0:
            torchvision.utils.save_image(x2_pos, os.path.join(exp_path, f'teacher_xpos2_epoch_{epoch}.png'), normalize=True)

        def _t_opt_state(o):
            return o.optimizer.state_dict() if hasattr(o, 'optimizer') else o.state_dict()

        if args.save_content and epoch % args.save_content_every == 0:
            print('[teacher] Saving content for resuming...')
            torch.save({
                'epoch': epoch + 1, 'global_step': global_step,
                't_gen_diff': t_gen_diff.state_dict(),
                't_gen_direct_f': t_gen_direct_f.state_dict(),
                't_gen_direct_r': t_gen_direct_r.state_dict(),
                't_disc_diff': t_disc_diff.state_dict(),
                't_disc_cycle': t_disc_cycle.state_dict(),
                'opt_t_diff': _t_opt_state(opt_t_diff),
                'opt_t_df': _t_opt_state(opt_t_df),
                'opt_t_dr': _t_opt_state(opt_t_dr),
                'opt_t_dd': opt_t_dd.state_dict(),
                'opt_t_dc': opt_t_dc.state_dict(),
                'sch_t_diff': sch_t_diff.state_dict(),
                'sch_t_df': sch_t_df.state_dict(),
                'sch_t_dr': sch_t_dr.state_dict(),
                'sch_t_dd': sch_t_dd.state_dict(),
                'sch_t_dc': sch_t_dc.state_dict(),
            }, teacher_content_file)

        # Save the teacher weights (after the EMA swap) for the student stage to load
        if epoch % args.save_ckpt_every == 0 or epoch == args.num_epoch:
            if args.use_ema:
                opt_t_diff.swap_parameters_with_ema(store_params_in_ema=True)
                opt_t_df.swap_parameters_with_ema(store_params_in_ema=True)
                opt_t_dr.swap_parameters_with_ema(store_params_in_ema=True)
            torch.save({
                'epoch': epoch,
                'teacher_gen_diff': t_gen_diff.state_dict(),
                'teacher_gen_direct_f': t_gen_direct_f.state_dict(),
                'teacher_gen_direct_r': t_gen_direct_r.state_dict(),
            }, os.path.join(exp_path, 'teacher_vfd.pth'))
            if args.use_ema:
                opt_t_diff.swap_parameters_with_ema(store_params_in_ema=True)
                opt_t_df.swap_parameters_with_ema(store_params_in_ema=True)
                opt_t_dr.swap_parameters_with_ema(store_params_in_ema=True)
            print(f'[teacher] Saved teacher_vfd.pth at epoch {epoch}')

        # Teacher-pipeline validation: tranA->tranB (with Eq. 8 low-frequency suppression)
        with torch.no_grad():
            psnr_vals = []
            for (a_val, b_val, _, m_val) in data_loader_val:
                src = a_val.to(device, non_blocking=True)
                real = b_val.to(device, non_blocking=True)
                msk = m_val.to(device, non_blocking=True)
                if weight_input:
                    src = mask_weighted_source(src, msk, args.vfd_teacher_gamma)
                x_init = torch.cat((torch.randn_like(real), src), axis=1)
                fake = sample_from_model(pos_coeff, t_gen_diff, args.num_timesteps,
                                         x_init, T, args, post_fn=lf_suppress)
                fake = torch.clamp(to_range_0_1(fake), 0, 1).cpu().numpy()
                real01 = torch.clamp(to_range_0_1(real), 0, 1).cpu().numpy()
                data_range = real01.max() - real01.min()
                if data_range > 0:
                    p = psnr(real01, fake, data_range=data_range)
                    psnr_vals.append(0 if np.isnan(p) else p)
            if psnr_vals:
                print(f'[teacher] Epoch {epoch} Val PSNR (tranA->tranB): {np.mean(psnr_vals):.2f} dB')

    print("[VFD-Diff teacher stage] Training finished; teacher weights saved to teacher_vfd.pth")

#%% Core training function (modified: add filenames when saving images)
def train_vfd_diff(args):
    from backbones.discriminator import Discriminator_small, Discriminator_large
    from backbones.ncsnpp_generator_adagn import NCSNpp
    import backbones.generator_resnet 
    from utils.EMA import EMA

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print(f'===== GPU configuration =====')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Using GPU index: {args.gpu_chose}')
    if torch.cuda.is_available():
        print(f'GPU name: {torch.cuda.get_device_name(args.gpu_chose)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(args.gpu_chose).total_memory / 1e9:.2f} GB')
    print(f'======================')
    
    device = torch.device(f'cuda:{args.gpu_chose}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = args.batch_size
    nz = args.nz

    accumulation_steps = args.accumulation_steps
    print(f"Gradient accumulation steps: {accumulation_steps}")

    # ----------------------
    # VFD-Diff configuration (toggles for the three contributions in the paper)
    # ----------------------
    use_vmg = not args.no_vmg   # vessel-mask-guided local enhancement (Eqs. 1-6)
    use_fde = not args.no_fde   # two-stage frequency-domain enhancement (Eqs. 7-8)
    use_dg = (not args.no_dg) and (args.vfd_stage == 'student')  # teacher-student distillation (Eq. 9)
    mask_needed = use_vmg or use_dg
    # FDE Eq. (7): forward high-frequency-enhanced noise
    noise_fn = (lambda x: hf_enhanced_noise_like(x, args.vfd_hf_lam, args.vfd_hf_k)) if use_fde else None
    # FDE Eq. (8): generation-side Fourier low-frequency suppression
    lf_suppress = (lambda x: fourier_low_freq_suppress(x, args.vfd_beta_d, args.vfd_cutoff)) if use_fde else None
    print(f'VFD-Diff: stage={args.vfd_stage} | VMG={use_vmg} | FDE={use_fde} | DG={use_dg}')

    # ----------------------
    # Data loading: use the modified dataset (returns filenames + vessel masks)
    # ----------------------
    dataset_train = CustomFFADataset(
        root_path=args.input_path,
        phase="train",
        contrast1=args.contrast1,
        contrast2=args.contrast2,
        compute_mask=mask_needed,
        frangi_dmax=args.vfd_frangi_dmax,
        frangi_dmin=args.vfd_frangi_dmin
    )
    dataset_val = CustomFFADataset(
        root_path=args.input_path,
        phase="val",
        contrast1=args.contrast1,
        contrast2=args.contrast2,
        compute_mask=mask_needed,
        frangi_dmax=args.vfd_frangi_dmax,
        frangi_dmin=args.vfd_frangi_dmin
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print(f'train data size: {len(data_loader_train)} batches ({len(dataset_train)} samples)')
    print(f'val data size: {len(data_loader_val)} batches ({len(dataset_val)} samples)')
    
    val_l1_loss = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])
    val_psnr_values = np.zeros([2, args.num_epoch + 1, len(data_loader_val)])
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # ----------------------
    # VFD-Diff teacher stage (paper Sec. 2.3): the teacher network is conditioned
    # on the vessel-mask-weighted image; a mirrored CFP->FFA pipeline trains
    # independently to convergence and saves weights for student-stage distillation
    # ----------------------
    if args.vfd_stage == 'teacher':
        train_teacher_stage(args, device, data_loader_train, data_loader_val,
                            use_vmg, noise_fn, lf_suppress, to_range_0_1)
        return

    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)
    # VFD-Diff DG: frozen teacher diffusion generator (same architecture as the
    # student gen_diffusive_2, conditioned on the vessel-mask-weighted image;
    # see the teacher-stage training)
    teacher_gen_diff = NCSNpp(args).to(device) if use_dg else None
    args.num_channels = 1
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[args.gpu_chose])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[args.gpu_chose])
    disc_diffusive_1 = Discriminator_large(
        nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)
    ).to(device)
    disc_diffusive_2 = Discriminator_large(
        nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)
    ).to(device)
    disc_non_diffusive_cycle1 = backbones.generator_resnet.define_D(gpu_ids=[args.gpu_chose])
    disc_non_diffusive_cycle2 = backbones.generator_resnet.define_D(gpu_ids=[args.gpu_chose])

    optimizer_disc_diffusive_1 = optim.Adam(disc_diffusive_1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_non_diffusive_1to2 = optim.Adam(gen_non_diffusive_1to2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_non_diffusive_2to1 = optim.Adam(gen_non_diffusive_2to1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle1 = optim.Adam(disc_non_diffusive_cycle1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle2 = optim.Adam(disc_non_diffusive_cycle2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_1to2 = EMA(optimizer_gen_non_diffusive_1to2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_2to1 = EMA(optimizer_gen_non_diffusive_2to1, ema_decay=args.ema_decay)

    # LR schedule: cosine (baseline) or paper (Sec. 3.3: constant first half + linear decay to 0 in the second half)
    scheduler_gen_diffusive_1 = make_lr_scheduler(optimizer_gen_diffusive_1, args)
    scheduler_gen_diffusive_2 = make_lr_scheduler(optimizer_gen_diffusive_2, args)
    scheduler_gen_non_diffusive_1to2 = make_lr_scheduler(optimizer_gen_non_diffusive_1to2, args)
    scheduler_gen_non_diffusive_2to1 = make_lr_scheduler(optimizer_gen_non_diffusive_2to1, args)
    scheduler_disc_diffusive_1 = make_lr_scheduler(optimizer_disc_diffusive_1, args)
    scheduler_disc_diffusive_2 = make_lr_scheduler(optimizer_disc_diffusive_2, args)
    scheduler_disc_non_diffusive_cycle1 = make_lr_scheduler(optimizer_disc_non_diffusive_cycle1, args)
    scheduler_disc_non_diffusive_cycle2 = make_lr_scheduler(optimizer_disc_non_diffusive_cycle2, args)

    exp = args.exp
    output_path = args.output_path
    exp_path = os.path.join(output_path, exp)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        repo_root = os.path.dirname(os.path.abspath(__file__))
        copy_source(__file__, exp_path)
        shutil.copytree(os.path.join(repo_root, 'backbones'), os.path.join(exp_path, 'backbones'), ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(os.path.join(repo_root, 'utils'), os.path.join(exp_path, 'utils'), ignore=shutil.ignore_patterns('__pycache__'))

    print("Initializing diffusion coefficients...")
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    print("Diffusion coefficients initialized successfully!")

    # VFD-Diff DG (paper Sec. 2.3): load the teacher trained in the teacher stage and freeze it
    if use_dg:
        teacher_ckpt = os.path.join(exp_path, 'teacher_vfd.pth')
        if not os.path.exists(teacher_ckpt):
            raise FileNotFoundError(
                f"Teacher weights not found at {teacher_ckpt}.\n"
                f"Please run the teacher stage first: python train.py --vfd_stage teacher ... (keep the other arguments identical)"
            )
        tstate = torch.load(teacher_ckpt, map_location=device)
        teacher_gen_diff.load_state_dict(tstate['teacher_gen_diff'])
        teacher_gen_diff.eval()
        for p in teacher_gen_diff.parameters():
            p.requires_grad = False
        print(f"=> VFD-Diff: loaded and froze the teacher network (teacher epoch {tstate.get('epoch', '?')})")

    global_step, epoch, init_epoch = 0, 0, 0
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=device)
            init_epoch = checkpoint['epoch']
            epoch = init_epoch
            gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
            gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
            gen_non_diffusive_1to2.load_state_dict(checkpoint['gen_non_diffusive_1to2_dict'])
            gen_non_diffusive_2to1.load_state_dict(checkpoint['gen_non_diffusive_2to1_dict'])
            optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
            scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
            optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
            scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2'])
            optimizer_gen_non_diffusive_1to2.load_state_dict(checkpoint['optimizer_gen_non_diffusive_1to2'])
            scheduler_gen_non_diffusive_1to2.load_state_dict(checkpoint['scheduler_gen_non_diffusive_1to2'])
            optimizer_gen_non_diffusive_2to1.load_state_dict(checkpoint['optimizer_gen_non_diffusive_2to1'])
            scheduler_gen_non_diffusive_2to1.load_state_dict(checkpoint['scheduler_gen_non_diffusive_2to1'])
            disc_diffusive_1.load_state_dict(checkpoint['disc_diffusive_1_dict'])
            optimizer_disc_diffusive_1.load_state_dict(checkpoint['optimizer_disc_diffusive_1'])
            scheduler_disc_diffusive_1.load_state_dict(checkpoint['scheduler_disc_diffusive_1'])
            disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
            optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
            scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])
            disc_non_diffusive_cycle1.load_state_dict(checkpoint['disc_non_diffusive_cycle1_dict'])
            optimizer_disc_non_diffusive_cycle1.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle1'])
            scheduler_disc_non_diffusive_cycle1.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle1'])
            disc_non_diffusive_cycle2.load_state_dict(checkpoint['disc_non_diffusive_cycle2_dict'])
            optimizer_disc_non_diffusive_cycle2.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle2'])
            scheduler_disc_non_diffusive_cycle2.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle2'])
            global_step = checkpoint['global_step']
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
            # VFD-Diff: restore the frozen teacher weights
            if use_dg and checkpoint.get('teacher_gen_diff_dict') is not None:
                teacher_gen_diff.load_state_dict(checkpoint['teacher_gen_diff_dict'])
                teacher_gen_diff.eval()
                print("=> loaded frozen teacher_gen_diff from checkpoint")

    for epoch in range(init_epoch, args.num_epoch + 1):
        gen_diffusive_1.train()
        gen_diffusive_2.train()
        gen_non_diffusive_1to2.train()
        gen_non_diffusive_2to1.train()
        disc_diffusive_1.train()
        disc_diffusive_2.train()
        disc_non_diffusive_cycle1.train()
        disc_non_diffusive_cycle2.train()

        # Modified: the data loader now returns four values (including the VFD-Diff vessel mask)
        for iteration, (x1, x2, filenames, vessel_mask) in enumerate(data_loader_train):
            vessel_mask = vessel_mask.to(device, non_blocking=True)
            for p in disc_diffusive_1.parameters():
                p.requires_grad = True
            for p in disc_diffusive_2.parameters():
                p.requires_grad = True
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = True
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = True

            if iteration % accumulation_steps == 0:
                disc_diffusive_1.zero_grad()
                disc_diffusive_2.zero_grad()
                disc_non_diffusive_cycle1.zero_grad()
                disc_non_diffusive_cycle2.zero_grad()

            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)

            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x1_t.requires_grad = True
            # VFD-Diff FDE (Eq. 7): forward noising in the target tranA->tranB direction uses high-frequency-enhanced noise
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2, noise_fn=noise_fn)
            x2_t.requires_grad = True

            D1_real = disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)
            D2_real = disc_diffusive_2(x2_t, t2, x2_tp1.detach()).view(-1)
            
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)
            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)
            
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)

            errD1_real = F.softplus(-D1_real).mean()
            errD2_real = F.softplus(-D2_real).mean()
            errD_real = (errD1_real + errD2_real) / accumulation_steps

            grad_penalty = 0.0
            if args.lazy_reg is None:
                grad1_real = torch.autograd.grad(outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
                grad1_penalty = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad2_real = torch.autograd.grad(outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = (args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty) / accumulation_steps
            else:
                if global_step % args.lazy_reg == 0:
                    grad1_real = torch.autograd.grad(outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
                    grad1_penalty = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad2_real = torch.autograd.grad(outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                    grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = (args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty) / accumulation_steps

            errD1_fake = F.softplus(output1).mean()
            errD2_fake = F.softplus(output2).mean()
            errD_fake = (errD1_fake + errD2_fake) / accumulation_steps

            errD = errD_real + errD_fake

            D_cycle1_real = disc_non_diffusive_cycle1(real_data1).view(-1)
            D_cycle2_real = disc_non_diffusive_cycle2(real_data2).view(-1)
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1)

            errD_cycle1_real = F.softplus(-D_cycle1_real).mean()
            errD_cycle2_real = F.softplus(-D_cycle2_real).mean()
            errD_cycle1_fake = F.softplus(D_cycle1_fake).mean()
            errD_cycle2_fake = F.softplus(D_cycle2_fake).mean()

            errD_cycle = ((errD_cycle1_real + errD_cycle2_real) + (errD_cycle1_fake + errD_cycle2_fake)) / accumulation_steps

            total_disc_loss = errD + grad_penalty + errD_cycle
            total_disc_loss.backward()

            if (iteration + 1) % accumulation_steps == 0:
                optimizer_disc_diffusive_1.step()
                optimizer_disc_diffusive_2.step()
                optimizer_disc_non_diffusive_cycle1.step()
                optimizer_disc_non_diffusive_cycle2.step()
                disc_diffusive_1.zero_grad()
                disc_diffusive_2.zero_grad()
                disc_non_diffusive_cycle1.zero_grad()
                disc_non_diffusive_cycle2.zero_grad()

            if iteration % 10 == 0:
                torch.cuda.empty_cache()

            for p in disc_diffusive_1.parameters():
                p.requires_grad = False
            for p in disc_diffusive_2.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = False

            if iteration % accumulation_steps == 0:
                gen_diffusive_1.zero_grad()
                gen_diffusive_2.zero_grad()
                gen_non_diffusive_1to2.zero_grad()
                gen_non_diffusive_2to1.zero_grad()

            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            # VFD-Diff FDE (Eq. 7): forward noising in the target direction uses high-frequency-enhanced noise
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2, noise_fn=noise_fn)
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)

            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict_cycle = gen_non_diffusive_1to2(x1_0_predict)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)
            x1_0_predict_cycle = gen_non_diffusive_2to1(x2_0_predict)

            # VFD-Diff DG (paper Sec. 2.3): the frozen teacher forwards with the
            # vessel-mask-weighted condition, sharing (x2_tp1, t2, latent_z2)
            # with the student; it runs before the student forward to reduce peak memory
            teacher_x2_0, teacher_feats = None, None
            if use_dg:
                with torch.no_grad():
                    teacher_cond = mask_weighted_source(
                        x1_0_predict.detach(), vessel_mask, args.vfd_teacher_gamma)
                    teacher_x2_0, teacher_feats = teacher_gen_diff(
                        torch.cat((x2_tp1.detach(), teacher_cond), axis=1), t2, latent_z2,
                        return_features=True)

            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            if use_dg:
                x2_0_predict_diff, student_feats = gen_diffusive_2(
                    torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2,
                    return_features=True)
            else:
                x2_0_predict_diff = gen_diffusive_2(
                    torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)

            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)
            errG1 = F.softplus(-output1).mean()
            errG2 = F.softplus(-output2).mean()
            errG_adv = (errG1 + errG2) / accumulation_steps

            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1)
            errG_cycle_adv1 = F.softplus(-D_cycle1_fake).mean()
            errG_cycle_adv2 = F.softplus(-D_cycle2_fake).mean()
            errG_cycle_adv = (errG_cycle_adv1 + errG_cycle_adv2) / accumulation_steps

            errG1_L1 = F.l1_loss(x1_0_predict_diff[:, [0], :], real_data1)
            # VFD-Diff VMG (Eq. 5): replace the L1 in the target direction (tranA->tranB) with vessel-mask-weighted L1
            if use_vmg:
                errG2_L1 = masked_l1_loss(x2_0_predict_diff[:, [0], :], real_data2,
                                          vessel_mask, args.vfd_lam_l1)
            else:
                errG2_L1 = F.l1_loss(x2_0_predict_diff[:, [0], :], real_data2)
            errG_L1 = (errG1_L1 + errG2_L1) / accumulation_steps

            # VFD-Diff VMG (Eq. 6): vessel-mask-weighted SSIM loss
            errG2_SSIM = torch.tensor(0.0, device=device)
            if use_vmg:
                errG2_SSIM = masked_ssim_loss(x2_0_predict_diff[:, [0], :], real_data2,
                                              vessel_mask, args.vfd_lam_ssim) / accumulation_steps

            # VFD-Diff DG: pixel-level distillation (reuses Eq. 5, aligning the student x0 to the teacher x0)
            # + feature-level distillation (Eq. 9)
            distill_pix = torch.tensor(0.0, device=device)
            distill_feat = torch.tensor(0.0, device=device)
            if use_dg:
                distill_pix = masked_l1_loss(x2_0_predict_diff[:, [0], :], teacher_x2_0[:, [0], :],
                                             vessel_mask, args.vfd_lam_l1) / accumulation_steps
                distill_feat = feature_distill_loss(student_feats, teacher_feats,
                                                    args.vfd_alpha, args.vfd_beta_mse) / accumulation_steps

            errG1_cycle = F.l1_loss(x1_0_predict_cycle, real_data1)
            errG2_cycle = F.l1_loss(x2_0_predict_cycle, real_data2)
            errG_cycle = (errG1_cycle + errG2_cycle) / accumulation_steps

            errG = args.lambda_l1_loss * errG_cycle + errG_adv + errG_cycle_adv + args.lambda_l1_loss * errG_L1 \
                 + args.vfd_ssim_weight * errG2_SSIM \
                 + args.vfd_distill_weight * (distill_pix + distill_feat)
            errG.backward()

            if (iteration + 1) % accumulation_steps == 0:
                optimizer_gen_diffusive_1.step()
                optimizer_gen_diffusive_2.step()
                optimizer_gen_non_diffusive_1to2.step()
                optimizer_gen_non_diffusive_2to1.step()
                for p in disc_diffusive_1.parameters():
                    p.requires_grad = True
                for p in disc_diffusive_2.parameters():
                    p.requires_grad = True
                for p in disc_non_diffusive_cycle1.parameters():
                    p.requires_grad = True
                for p in disc_non_diffusive_cycle2.parameters():
                    p.requires_grad = True

            global_step += 1

            if iteration % 100 == 0:
                print(
                    f'epoch {epoch} iteration{iteration}, '
                    f'G-Cycle: {errG_cycle.item() * accumulation_steps:.4f}, '
                    f'G-L1: {errG_L1.item() * accumulation_steps:.4f}, '
                    f'G-Adv: {errG_adv.item() * accumulation_steps:.4f}, '
                    f'G-cycle-Adv: {errG_cycle_adv.item() * accumulation_steps:.4f}, '
                    f'G-SSIM: {errG2_SSIM.item() * accumulation_steps:.4f}, '
                    f'G-Distill: {(distill_pix + distill_feat).item() * accumulation_steps:.4f}, '
                    f'G-Sum: {errG.item() * accumulation_steps:.4f}, '
                    f'D Loss: {errD.item() * accumulation_steps:.4f}, '
                    f'D_cycle Loss: {errD_cycle.item() * accumulation_steps:.4f}'
                )

        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_gen_non_diffusive_1to2.step()
            scheduler_gen_non_diffusive_2to1.step()
            scheduler_disc_diffusive_1.step()
            scheduler_disc_diffusive_2.step()
            scheduler_disc_non_diffusive_cycle1.step()
            scheduler_disc_non_diffusive_cycle2.step()

        if epoch % 10 == 0:
            torchvision.utils.save_image(x1_pos_sample, os.path.join(exp_path, f'xpos1_epoch_{epoch}.png'), normalize=True)
            torchvision.utils.save_image(x2_pos_sample, os.path.join(exp_path, f'xpos2_epoch_{epoch}.png'), normalize=True)

        with torch.no_grad():
            # ----------------------
            # Modified: generate the sample2_discrete_epoch image with filenames added
            # ----------------------
            # Save the original sample1_discrete (unchanged)
            x1_t = torch.cat((torch.randn_like(real_data1), real_data2), axis=1)
            fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
            fake_sample1 = torch.cat((real_data2, fake_sample1), axis=-1)
            torchvision.utils.save_image(fake_sample1, os.path.join(exp_path, f'sample1_discrete_epoch_{epoch}.png'), normalize=True)

            # Save the original sample1_translated (unchanged)
            pred1 = gen_non_diffusive_2to1(real_data2)
            x2_t = torch.cat((torch.randn_like(real_data2), pred1), axis=1)
            fake_sample2_tilda = gen_diffusive_2(x2_t, t2, latent_z2)
            pred1 = torch.cat((real_data2, pred1, gen_non_diffusive_1to2(pred1), fake_sample2_tilda[:, [0], :]), axis=-1)
            torchvision.utils.save_image(pred1, os.path.join(exp_path, f'sample1_translated_epoch_{epoch}.png'), normalize=True)

            # ----------------------
            # Core modification: sample2_discrete_epoch - add filenames
            # ----------------------
            x2_t = torch.cat((torch.randn_like(real_data2), real_data1), axis=1)
            fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
            
            # Concatenate: real_data1 (tranA) + fake_sample2 (generated tranB) + real_data2 (real tranB)
            concatenated = torch.cat((real_data1, fake_sample2, real_data2), axis=-1)
            
            # Add the filename to each image
            images_with_filenames = []
            for i in range(concatenated.shape[0]):
                img_with_name = add_filename_to_image(concatenated[i], filenames[i])
                images_with_filenames.append(img_with_name)
            
            images_with_filenames = torch.stack(images_with_filenames)
            
            # Save the images with filenames added
            torchvision.utils.save_image(
                images_with_filenames, 
                os.path.join(exp_path, f'sample2_discrete_epoch_{epoch}.png'), 
                normalize=True,
                nrow=1  # one image per row so the filenames stay clearly visible
            )

            # Save the original sample2_translated (unchanged)
            pred2 = gen_non_diffusive_1to2(real_data1)
            x1_t = torch.cat((torch.randn_like(real_data1), pred2), axis=1)
            fake_sample1_tilda = gen_diffusive_1(x1_t, t1, latent_z1)
            pred2 = torch.cat((real_data1, pred2, gen_non_diffusive_2to1(pred2), fake_sample1_tilda[:, [0], :]), axis=-1)
            torchvision.utils.save_image(pred2, os.path.join(exp_path, f'sample2_translated_epoch_{epoch}.png'), normalize=True)

        def get_optimizer_state(optimizer):
            if hasattr(optimizer, 'optimizer'):
                return optimizer.optimizer.state_dict()
            else:
                return optimizer.state_dict()

        if args.save_content:
            if epoch % args.save_content_every == 0:
                print('Saving content for resuming...')
                content = {
                    'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                    'gen_diffusive_1_dict': gen_diffusive_1.state_dict(),
                    'optimizer_gen_diffusive_1': get_optimizer_state(optimizer_gen_diffusive_1),
                    'gen_diffusive_2_dict': gen_diffusive_2.state_dict(),
                    'optimizer_gen_diffusive_2': get_optimizer_state(optimizer_gen_diffusive_2),
                    'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(),
                    'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(),
                    'gen_non_diffusive_1to2_dict': gen_non_diffusive_1to2.state_dict(),
                    'optimizer_gen_non_diffusive_1to2': get_optimizer_state(optimizer_gen_non_diffusive_1to2),
                    'gen_non_diffusive_2to1_dict': gen_non_diffusive_2to1.state_dict(),
                    'optimizer_gen_non_diffusive_2to1': get_optimizer_state(optimizer_gen_non_diffusive_2to1),
                    'scheduler_gen_non_diffusive_1to2': scheduler_gen_non_diffusive_1to2.state_dict(),
                    'scheduler_gen_non_diffusive_2to1': scheduler_gen_non_diffusive_2to1.state_dict(),
                    'disc_diffusive_1_dict': disc_diffusive_1.state_dict(),
                    'optimizer_disc_diffusive_1': optimizer_disc_diffusive_1.state_dict(),
                    'scheduler_disc_diffusive_1': scheduler_disc_diffusive_1.state_dict(),
                    'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                    'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(),
                    'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                    'disc_non_diffusive_cycle1_dict': disc_non_diffusive_cycle1.state_dict(),
                    'optimizer_disc_non_diffusive_cycle1': optimizer_disc_non_diffusive_cycle1.state_dict(),
                    'scheduler_disc_non_diffusive_cycle1': scheduler_disc_non_diffusive_cycle1.state_dict(),
                    'disc_non_diffusive_cycle2_dict': disc_non_diffusive_cycle2.state_dict(),
                    'optimizer_disc_non_diffusive_cycle2': optimizer_disc_non_diffusive_cycle2.state_dict(),
                    'scheduler_disc_non_diffusive_cycle2': scheduler_disc_non_diffusive_cycle2.state_dict(),
                    # VFD-Diff: frozen teacher state (for resuming the student stage)
                    'teacher_gen_diff_dict': teacher_gen_diff.state_dict() if use_dg else None,
                }
                torch.save(content, os.path.join(exp_path, 'content.pth'))

        if epoch % args.save_ckpt_every == 0 or epoch == args.num_epoch:
            if args.use_ema:
                optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)
            torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, f'gen_diffusive_1_{epoch}.pth'))
            torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, f'gen_diffusive_2_{epoch}.pth'))
            torch.save(gen_non_diffusive_1to2.state_dict(), os.path.join(exp_path, f'gen_non_diffusive_1to2_{epoch}.pth'))
            torch.save(gen_non_diffusive_2to1.state_dict(), os.path.join(exp_path, f'gen_non_diffusive_2to1_{epoch}.pth'))
            if args.use_ema:
                optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)
            print(f'Saved checkpoint at epoch {epoch}')

        with torch.no_grad():
            # Direction-1 validation: tranB->tranA (gen_diffusive_1)
            for iteration, (a_val, b_val, _, _) in enumerate(data_loader_val):
                real_data = a_val.to(device, non_blocking=True)
                source_data = b_val.to(device, non_blocking=True)
                x1_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
                fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
                fake_sample1 = to_range_0_1(fake_sample1)
                real_data = to_range_0_1(real_data)
                fake_sample1 = torch.clamp(fake_sample1, 0, 1)
                real_data = torch.clamp(real_data, 0, 1)
                fake_sample1 = fake_sample1.cpu().numpy()
                real_data = real_data.cpu().numpy()
                l1 = np.abs(fake_sample1 - real_data).mean()
                val_l1_loss[0, epoch, iteration] = l1 if not np.isnan(l1) else 0
                data_range = real_data.max() - real_data.min()
                psnr_val = psnr(real_data, fake_sample1, data_range=data_range) if data_range > 0 else 0
                val_psnr_values[0, epoch, iteration] = psnr_val if not np.isnan(psnr_val) else 0

            # Direction-2 validation: tranA->tranB (gen_diffusive_2, target direction)
            # Fix: the original baseline loop mistakenly used gen_diffusive_1 here;
            # it now matches test.py
            # VFD-Diff FDE (Eq. 8): generated images go through Fourier low-frequency
            # suppression before computing metrics
            for iteration, (a_val, b_val, _, _) in enumerate(data_loader_val):
                source_data = a_val.to(device, non_blocking=True)   # tranA source
                real_data = b_val.to(device, non_blocking=True)     # tranB target
                x2_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
                fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps,
                                                 x2_t, T, args, post_fn=lf_suppress)
                fake_sample2 = to_range_0_1(fake_sample2)
                real_data = to_range_0_1(real_data)
                fake_sample2 = torch.clamp(fake_sample2, 0, 1)
                real_data = torch.clamp(real_data, 0, 1)
                fake_sample2 = fake_sample2.cpu().numpy()
                real_data = real_data.cpu().numpy()
                l1 = np.abs(fake_sample2 - real_data).mean()
                val_l1_loss[1, epoch, iteration] = l1 if not np.isnan(l1) else 0
                data_range = real_data.max() - real_data.min()
                psnr_val = psnr(real_data, fake_sample2, data_range=data_range) if data_range > 0 else 0
                val_psnr_values[1, epoch, iteration] = psnr_val if not np.isnan(psnr_val) else 0

        mean_psnr_1 = np.nanmean(val_psnr_values[0, epoch, :])
        mean_psnr_2 = np.nanmean(val_psnr_values[1, epoch, :])
        mean_l1_1 = np.nanmean(val_l1_loss[0, epoch, :])
        mean_l1_2 = np.nanmean(val_l1_loss[1, epoch, :])
        print(f'Epoch {epoch} Val PSNR (tranB->tranA): {mean_psnr_1:.2f} dB | L1: {mean_l1_1:.4f}')
        print(f'Epoch {epoch} Val PSNR (tranA->tranB, target direction): {mean_psnr_2:.2f} dB | L1: {mean_l1_2:.4f}')

        np.save(os.path.join(exp_path, 'val_l1_loss.npy'), val_l1_loss)
        np.save(os.path.join(exp_path, 'val_psnr_values.npy'), val_psnr_values)

#%% Main entry (unchanged)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('VFD-Diff parameters')
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--image_size', type=int, default=512, help='size of image')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False, help='geometric noise schedule')
    parser.add_argument('--beta_min', type=float, default=0.1, help='diffusion beta_min')
    parser.add_argument('--beta_max', type=float, default=20., help='diffusion beta_max')

    parser.add_argument('--num_channels_dae', type=int, default=128, help='initial number of channels of the denoising model')
    parser.add_argument('--n_mlp', type=int, default=3, help='number of MLP layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, required=True, help='channel multipliers')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of ResNet blocks per scale')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(64, 32, 16), help='resolutions at which attention is applied')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='use conv for up/downsampling')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditioning')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR filtering')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='ResNet block type')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='output progression type')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='input progression type')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progression combination mode')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='time embedding type')
    parser.add_argument('--fourier_scale', type=float, default=16., help='Fourier transform scale')
    parser.add_argument('--not_use_tanh', action='store_true', default=False, help='whether to use tanh')

    parser.add_argument('--exp', default='exp_ffa_generation', help='experiment name')
    parser.add_argument('--input_path', required=True, help='input data path')
    parser.add_argument('--output_path', required=True, help='output save path')
    parser.add_argument('--nz', type=int, default=100, help='latent dimension')
    parser.add_argument('--num_timesteps', type=int, default=4, help='diffusion timesteps')
    parser.add_argument('--z_emb_dim', type=int, default=512, help='z embedding dimension')
    parser.add_argument('--t_emb_dim', type=int, default=512, help='t embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epoch', type=int, default=1200, help='number of training epochs')
    parser.add_argument('--ngf', type=int, default=64, help='number of discriminator filters')
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='adam beta2')
    parser.add_argument('--no_lr_decay', action='store_true', default=False, help='disable learning-rate decay')
    parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='R1 regularization coefficient')
    parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regularization')
    parser.add_argument('--save_content', action='store_true', default=False, help='save training state')
    parser.add_argument('--save_content_every', type=int, default=10, help='save training state every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save model weights every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5, help='L1 loss weight')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='gradient accumulation steps')

    parser.add_argument('--contrast1', type=str, default='tranA', help='source modality root folder')
    parser.add_argument('--contrast2', type=str, default='tranB', help='target modality root folder')

    parser.add_argument('--gpu_chose', type=int, default=0, help='GPU index to use')

    # ---------------- VFD-Diff parameters (the three contributions in the paper) ----------------
    parser.add_argument('--vfd_stage', type=str, default='student', choices=['student', 'teacher'],
                        help='teacher=independent teacher-stage training; student=student stage (baseline training + distillation, requires the teacher stage to finish first)')
    # Ablation switches (corresponding to Tables 1/2 in the paper)
    parser.add_argument('--no_vmg', action='store_true', default=False, help='disable VMG (vessel-mask-guided local enhancement)')
    parser.add_argument('--no_fde', action='store_true', default=False, help='disable FDE (two-stage frequency-domain enhancement)')
    parser.add_argument('--no_dg', action='store_true', default=False, help='disable DG (teacher-student distillation)')
    # VMG (Eqs. 5/6): the paper gives no values; the following are default settings
    parser.add_argument('--vfd_lam_l1', type=float, default=5.0, help='vessel-mask weight lambda_L1 in Eq. (5)')
    parser.add_argument('--vfd_lam_ssim', type=float, default=5.0, help='vessel-mask weight lambda_SSIM in Eq. (6)')
    parser.add_argument('--vfd_ssim_weight', type=float, default=1.0, help='weight of the weighted-SSIM term in the total loss')
    # FDE (Eqs. 7/8)
    parser.add_argument('--vfd_hf_lam', type=float, default=0.5, help='high-frequency enhancement strength lambda in Eq. (7)')
    parser.add_argument('--vfd_hf_k', type=int, default=5, help='high-pass Gaussian kernel size in Eq. (7)')
    parser.add_argument('--vfd_beta_d', type=float, default=0.1, help='low-frequency decay coefficient beta_d in Eq. (8)')
    parser.add_argument('--vfd_cutoff', type=int, default=32, help='low-frequency mask radius in Eq. (8) (pixels)')
    # DG (Eq. 9)
    parser.add_argument('--vfd_alpha', type=float, default=1.0, help='cosine-similarity term weight alpha in Eq. (9)')
    parser.add_argument('--vfd_beta_mse', type=float, default=1.0, help='MSE term weight beta in Eq. (9)')
    parser.add_argument('--vfd_distill_weight', type=float, default=1.0, help='total weight of the distillation loss (pixel + feature)')
    parser.add_argument('--vfd_teacher_gamma', type=float, default=2.0, help='vessel weighting coefficient gamma for the teacher input (w = 1 + gamma * M)')
    # Adaptive Frangi (Eqs. 2-4)
    parser.add_argument('--vfd_frangi_dmax', type=float, default=16.0, help='max vessel diameter D_max in Eqs. (2)(4) (pixels, 512-based)')
    parser.add_argument('--vfd_frangi_dmin', type=float, default=2.0, help='min vessel diameter D_min in Eq. (4) (pixels, 512-based)')
    # LR schedule (paper Sec. 3.3: constant 1.5e-4 for the first 300 epochs, then linear decay to 0 over the last 300)
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'paper'],
                        help="cosine=baseline cosine annealing; paper=paper protocol (constant first half + linear decay in the second half)")

    args = parser.parse_args()
    train_vfd_diff(args)
