# -*- coding: utf-8 -*-
"""
VFD-Diff 核心模块(论文公式的代码实现)
论文: "Vessel-aware frequency distillation diffusion with a teacher-student
       network for fundus fluorescein angiography synthesis"

公式编号与论文一致:
  公式 (1)-(4): adaptive_frangi_mask       参数自适应 Frangi 血管掩码 M_VSA
  公式 (5)    : masked_l1_loss             血管掩码加权 L1 损失
  公式 (6)    : masked_ssim_loss           血管掩码加权 SSIM 损失
  公式 (7)    : hf_enhanced_noise_like     前向扩散高频增强噪声
  公式 (8)    : fourier_low_freq_suppress  生成端傅里叶低频抑制
  公式 (9)    : feature_distill_loss       特征级蒸馏损失(余弦相似度+MSE)
辅助函数: gaussian_blur / mask_weighted_source(教师网络输入构造)
"""
import math
import torch
import torch.nn.functional as F

# =====================================================================
# 高斯滤波工具(固定核,不参与梯度)
# =====================================================================

def _gaussian_kernel1d(sigma, radius):
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _sep_gaussian_blur(x, kernel1d):
    """可分离高斯模糊。x: [B,C,H,W], kernel1d: 1D核(长度为奇数)"""
    C = x.shape[1]
    r = len(kernel1d) // 2
    kernel1d = kernel1d.to(device=x.device, dtype=x.dtype)
    kx = kernel1d.view(1, 1, 1, -1).expand(C, 1, 1, len(kernel1d))
    ky = kernel1d.view(1, 1, -1, 1).expand(C, 1, len(kernel1d), 1)
    x = F.conv2d(x, kx, padding=(0, r), groups=C)
    x = F.conv2d(x, ky, padding=(r, 0), groups=C)
    return x


# =====================================================================
# 公式 (1)-(4): 参数自适应 Frangi 血管掩码
# =====================================================================

def _hist_peak_count(flat, bins=64, min_rel=0.05):
    """灰度直方图显著峰数 N_peak(平滑后计数局部极大)"""
    hist = torch.histc(flat.clamp(0, 1), bins=bins, min=0.0, max=1.0)
    k = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32).view(1, 1, -1)
    hs = F.conv1d(hist.view(1, 1, -1), k, padding=1).view(-1)
    thr = hs.max().item() * min_rel
    if hs.numel() < 3:
        return 1
    peaks = (hs[1:-1] > hs[:-2]) & (hs[1:-1] > hs[2:]) & (hs[1:-1] > thr)
    return max(1, int(peaks.sum().item()))


def _mean_local_entropy(x, levels=16, block=16):
    """平均局部熵 H_local,归一化到 [0,1](非重叠 block×block 块的灰度熵均值)"""
    q = (x.clamp(0, 1) * (levels - 1)).round().long()[0, 0]  # [H,W]
    Hh, Ww = q.shape
    hb, wb = Hh // block, Ww // block
    if hb == 0 or wb == 0:
        return 0.0
    q = q[:hb * block, :wb * block].view(hb, block, wb, block)
    q = q.permute(0, 2, 1, 3).reshape(hb * wb, block * block)
    hist = torch.stack([(q == v).float().mean(dim=1) for v in range(levels)], dim=1)
    hist = hist.clamp(min=1e-12)
    ent = -(hist * hist.log2()).sum(dim=1) / math.log2(levels)
    return ent.mean().item()


def _hessian_components(x, sigma):
    """尺度归一化的 Hessian 分量 Lxx/Lyy/Lxy(高斯二阶导数,scale-normalized)"""
    radius = int(math.ceil(3.0 * sigma))
    i = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(i ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()                                   # 归一化高斯
    dg = -i / (sigma ** 2) * g                        # 一阶导核
    d2g = ((i ** 2 - sigma ** 2) / (sigma ** 4)) * g  # 二阶导核
    # 2D 核 = 行/列方向 1D 核外积;再乘 sigma^2 做尺度归一化
    kxx = (g.view(-1, 1) * d2g.view(1, -1)) * (sigma ** 2)   # 沿x二阶导,沿y平滑
    kyy = (d2g.view(-1, 1) * g.view(1, -1)) * (sigma ** 2)
    kxy = (dg.view(-1, 1) * dg.view(1, -1)) * (sigma ** 2)
    pad = radius
    lxx = F.conv2d(x, kxx.view(1, 1, 2 * pad + 1, 2 * pad + 1), padding=pad)
    lyy = F.conv2d(x, kyy.view(1, 1, 2 * pad + 1, 2 * pad + 1), padding=pad)
    lxy = F.conv2d(x, kxy.view(1, 1, 2 * pad + 1, 2 * pad + 1), padding=pad)
    return lxx, lyy, lxy


@torch.no_grad()
def adaptive_frangi_mask(img01, invert=False, d_max=16.0, d_min=2.0,
                         dilate_iters=2, thresh_pct=88.0):
    """
    参数自适应 Frangi 血管掩码(论文公式 1-4 + 阈值/膨胀后处理)。
    img01: float tensor [1,1,H,W],取值[0,1](CFP 眼底照,血管为暗管状,勿需反转;
           FFA 亮血管请传 invert=True)
    返回: [1,1,H,W] 二值掩码 M_VSA(0./1.)
    """
    x = (1.0 - img01) if invert else img01
    H, W = x.shape[-2], x.shape[-1]
    scale = (H / 512.0 + W / 512.0) / 2.0            # D_max/D_min 以 512×512 为基准
    d_max, d_min = d_max * scale, d_min * scale

    # ---- 图像统计特征 → 公式(2)(3)(4) 的自适应参数 ----
    flat = x.flatten()
    p1 = torch.quantile(flat, 0.01)
    p99 = torch.quantile(flat, 0.99)
    c_global = float(((p99 - p1) / 2.0).clamp(0.0, 0.5))    # 全局对比度
    n_peak = _hist_peak_count(flat)                          # 直方图显著峰数
    h_local = _mean_local_entropy(x)                         # 平均局部熵(归一化)

    s_max = d_max / (2.0 * math.sqrt(3.0)) * (1.0 + c_global)          # 公式(2)
    n_scales = int(max(4, min(12, n_peak + 2)))                        # 公式(3)
    beta = 0.5 * (d_max - d_min) / (d_max + d_min) * (1.0 + h_local)   # 公式(4)
    beta = float(min(max(beta, 0.1), 1.5))

    scales = torch.linspace(max(1.0, s_max / n_scales), max(2.0, s_max), n_scales)
    response = torch.zeros_like(x)
    for s in scales.tolist():
        lxx, lyy, lxy = _hessian_components(x, s)
        tr = lxx + lyy
        det_sqrt = torch.sqrt(((lxx - lyy) / 2.0) ** 2 + lxy ** 2 + 1e-12)
        lam_a = tr / 2.0 - det_sqrt          # 较小特征值
        lam_b = tr / 2.0 + det_sqrt          # 较大特征值
        # 按 |λ1| <= |λ2| 排序(管状结构参数 R_B = λ1/λ2 的定义要求)
        swap = lam_a.abs() > lam_b.abs()
        lam1 = torch.where(swap, lam_b, lam_a)
        lam2 = torch.where(swap, lam_a, lam_b)
        rb = (lam1 / (lam2 + 1e-12)).clamp(-1.0, 1.0)     # 管状结构参数 R_B
        S = lam1 ** 2 + lam2 ** 2                          # 结构显著度 S
        c = 0.5 * S.max().clamp(min=1e-6)                  # Frangi 标准: 本尺度最大显著度一半
        vessel = torch.exp(-(rb ** 2) / (2.0 * beta ** 2)) * \
                 (1.0 - torch.exp(-S / (2.0 * c ** 2)))    # 公式(1)
        # 论文公式(1): λ2 < 0 时响应置 0(暗管状结构,对应 CFP 中的暗血管)
        vessel = torch.where(lam2 > 0, vessel, torch.zeros_like(vessel))
        response = torch.maximum(response, vessel)

    pos = response[response > 1e-6]
    if pos.numel() == 0 or float(response.max()) < 1e-4:
        return torch.zeros_like(x)

    # 阈值 + 形态学膨胀(3×3 最大池化)→ 二值血管掩码 M_VSA
    thr = max(0.05, float(torch.quantile(pos, thresh_pct / 100.0)))
    mask = (response > thr).float()
    for _ in range(dilate_iters):
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return mask


# =====================================================================
# 公式 (7): 前向扩散高频增强噪声
# =====================================================================

def hf_enhanced_noise_like(x, lam=0.5, k=5):
    """n̄ = n + λ·HP(n)。HP 为高斯差分高通(n − GaussBlur(n))。返回与 x 同形状噪声。"""
    n = torch.randn_like(x)
    g = _gaussian_kernel1d(max(1.0, k / 4.0), k // 2)
    blur = _sep_gaussian_blur(n, g)
    return n + lam * (n - blur)


# =====================================================================
# 公式 (8): 生成端傅里叶低频抑制
# =====================================================================

def fourier_low_freq_suppress(x, beta_d=0.1, cutoff=32):
    """x = F⁻¹( F(x) ⊙ (1 − β_d·M_LP) )。x: [B,C,H,W];cutoff 为低频半径(像素)"""
    B, C, H, W = x.shape
    spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    yy = torch.arange(H, device=x.device, dtype=torch.float32).view(-1, 1) - H // 2
    xx = torch.arange(W, device=x.device, dtype=torch.float32).view(1, -1) - W // 2
    r2 = yy ** 2 + xx ** 2
    m_lp = (r2 <= float(cutoff) ** 2).to(dtype=x.dtype)     # 低通掩码 M_LP
    out_spec = torch.fft.ifftshift(spec * (1.0 - beta_d * m_lp), dim=(-2, -1))
    return torch.fft.ifft2(out_spec).real


# =====================================================================
# 公式 (5)(6): 血管掩码加权 L1 / SSIM 损失
# =====================================================================

def masked_l1_loss(gen, gt, mask, lam=5.0):
    """公式(5): L = E[ |I_gen − I_gt| ⊙ (1 + (λ_L1−1)·M_VSA) ]"""
    w = 1.0 + (lam - 1.0) * mask
    return ((gen - gt).abs() * w).mean()


def _ssim_map(x, y, data_range=2.0, win=11, sigma=1.5):
    """标准 SSIM 图(11×11 高斯窗)。x/y: [B,C,H,W],取值范围 [-data_range/2, data_range/2]"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    g = _gaussian_kernel1d(sigma, win // 2).to(device=x.device, dtype=x.dtype)

    def blur(t):
        return _sep_gaussian_blur(t, g)

    mx, my = blur(x), blur(y)
    mxx, myy, mxy = blur(x * x), blur(y * y), blur(x * y)
    vx = (mxx - mx * mx).clamp(min=0.0)
    vy = (myy - my * my).clamp(min=0.0)
    vxy = mxy - mx * my
    return ((2.0 * mx * my + C1) * (2.0 * vxy + C2)) / \
           ((mx * mx + my * my + C1) * (vx + vy + C2))


def masked_ssim_loss(gen, gt, mask, lam=5.0, data_range=2.0):
    """公式(6): L = 1 − E[ S(I_gen, I_gt) ⊙ (1 + (λ_SSIM−1)·M_VSA) ]"""
    w = 1.0 + (lam - 1.0) * mask
    return 1.0 - (_ssim_map(gen, gt, data_range=data_range) * w).mean()


# =====================================================================
# 公式 (9): 特征级蒸馏损失(通道方向余弦相似度 + MSE)
# =====================================================================

def feature_distill_loss(feats_s, feats_t, alpha=1.0, beta_mse=1.0):
    """公式(9): L_structure = α·(1−CSLoss) + β·MSE,逐层计算后取均值。
    feats_*: NCSNpp 各尺度 ResNet 块输出特征列表(师生同构,层一一对应)"""
    assert len(feats_s) == len(feats_t) and len(feats_s) > 0
    total = 0.0
    for fs, ft in zip(feats_s, feats_t):
        fs = fs.flatten(2)                       # [B,C,N]
        ft = ft.detach().flatten(2)
        cs = F.cosine_similarity(fs, ft, dim=1).mean()   # 通道方向余弦相似度
        mse = F.mse_loss(fs, ft)
        total = total + alpha * (1.0 - cs) + beta_mse * mse
    return total / len(feats_s)


# =====================================================================
# 教师网络输入构造
# =====================================================================

def mask_weighted_source(x, mask, gamma=2.0):
    """血管掩码加权图(教师网络输入):w = 1+γ·M_VSA,加权后逐样本 renorm 到 [-1,1]"""
    xw = x * (1.0 + gamma * mask)
    mn = xw.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
    mx = xw.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    return 2.0 * (xw - mn) / (mx - mn + 1e-8) - 1.0
