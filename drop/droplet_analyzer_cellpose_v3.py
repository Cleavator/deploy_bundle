#!/usr/bin/env python3
# droplet_analyzer_cellpose_v3.py
"""
高性能 + 高鲁棒 + 易扩展
"""
import os
import logging
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from cellpose import models, io
from skimage import measure, color
from skimage.segmentation import mark_boundaries
import cv2
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- 0. Helper: Device & Model ---------- #
def get_device_and_gpu_flag():
    """
    Automatic device selection: CUDA > MPS > CPU
    Returns: (device_object, gpu_bool)
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    elif torch.backends.mps.is_available():
        return torch.device("mps"), False  # Cellpose's 'gpu' arg is mainly for CUDA
    else:
        return torch.device("cpu"), False

_CP_MODEL_CACHE = {}
def load_cellpose_model_robust(model_name="cyto3"):
    """
    Robust Cellpose model loading for v3/v4 compatibility.
    """
    device, gpu_flag = get_device_and_gpu_flag()
    logger.info(f"Using device: {device}, model: {model_name}")
    
    cache_key = f"{model_name}|{device}"
    if cache_key in _CP_MODEL_CACHE:
        return _CP_MODEL_CACHE[cache_key]
    try:
        if model_name != "cyto3":
            raise RuntimeError("Only cyto3 is allowed")
        model = models.CellposeModel(pretrained_model=model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to load Cellpose model: {e}")
        raise e
        
    _CP_MODEL_CACHE[cache_key] = model
    logger.info("Cellpose model initialized")
    return _CP_MODEL_CACHE[cache_key]

# ---------- 0.5 Helper: Auto Target Detection ---------- #
def infer_target_from_image(img, q=0.95, thr=0.10):
    """
    Infer target (miR-21/miR-92a) based on Red/Green channel dominance.
    
    Args:
        img: np.ndarray (H, W, C) or (H, W)
        q: quantile for top pixels (default 0.95)
        thr: score threshold (default 0.10)
        
    Returns:
        (target, score, R_top, G_top)
        target: "miR-21" (Red), "miR-92a" (Green), or "ambiguous"
    """
    # 1. 维度检查
    if img.ndim == 2:
        # Grayscale -> Ambiguous
        return "ambiguous", 0.0, 0.0, 0.0
        
    if img.ndim == 3:
        if img.shape[-1] < 3:
            # Not enough channels
            return "ambiguous", 0.0, 0.0, 0.0
        # Ignore Alpha if RGBA
        if img.shape[-1] == 4:
            img = img[..., :3]
            
    # 2. 位深归一化 (16-bit to 8-bit scale float)
    # 无论是 uint8 还是 uint16，我们都统一转为 float32 处理
    img_float = img.astype(np.float32)
    
    # 如果是 16-bit (max > 255)，归一化到 0-255
    if img.dtype != np.uint8 and img.max() > 255:
        # 避免除以0
        max_val = img.max()
        if max_val > 0:
            img_float = (img_float / max_val) * 255.0
        
    # Extract R and G channels (assuming RGB)
    R = img_float[..., 0]
    G = img_float[..., 1]
    
    # Calculate top quantile mean
    try:
        r_thresh = np.quantile(R, q)
        g_thresh = np.quantile(G, q)
        
        R_top = R[R >= r_thresh].mean()
        G_top = G[G >= g_thresh].mean()
        
        # Calculate score: (G - R) / (G + R)
        denom = G_top + R_top + 1e-8
        score = (G_top - R_top) / denom
        
        if score > thr:
            return "miR-92a", score, R_top, G_top
        elif score < -thr:
            return "miR-21", score, R_top, G_top
        else:
            return "ambiguous", score, R_top, G_top
            
    except Exception as e:
        logger.warning(f"Error in auto target detection: {e}")
        return "ambiguous", None, None, None

# ---------- 1. 数据集 ---------- #
def scan_dataset(root):
    """返回 (path, label_idx, id_name) 列表 + 类别表"""
    items = []
    IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    # 0. 单文件
    if os.path.isfile(root) and os.path.splitext(root)[1].lower() in IMG_EXT:
        return [(root, 0, "default")], {"default": 0}, {"default": "default"}

    # 1. 两级子目录 root/cls/id/*.tif
    cls_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if cls_dirs and any(
        os.path.isdir(os.path.join(root, cls_dirs[0], sub)) for sub in os.listdir(os.path.join(root, cls_dirs[0]))
    ):
        classes = sorted(cls_dirs)
        cls2idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            cls_path = os.path.join(root, cls)
            for id_name in os.listdir(cls_path):
                id_path = os.path.join(cls_path, id_name)
                if not os.path.isdir(id_path):
                    continue
                for f in os.listdir(id_path):
                    if os.path.splitext(f)[1].lower() in IMG_EXT:
                        items.append((os.path.join(id_path, f), cls2idx[cls], id_name))
        return items, cls2idx, {id_: cls_idx for _, cls_idx, id_ in items}

    # 2. 一级子目录 root/cls/*.tif
    if cls_dirs:
        classes = sorted(cls_dirs)
        cls2idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            cls_path = os.path.join(root, cls)
            for f in os.listdir(cls_path):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    items.append((os.path.join(cls_path, f), cls2idx[cls], "default"))
        return items, cls2idx, {id_: cls_idx for _, cls_idx, id_ in items}

    # 3. 平铺文件 root/*.tif
    files = [f for f in os.listdir(root) if os.path.splitext(f)[1].lower() in IMG_EXT]
    if files:
        return (
            [(os.path.join(root, f), 0, "default") for f in files],
            {"default": 0},
            {"default": "default"},
        )

    raise ValueError(f"No images found under {root}")


def normalize_image_to_uint8(image):
    """
    将图像转换为灰度图并归一化到0-255范围(uint8)
    """
    # 转换为灰度图
    if image.ndim == 3 and image.shape[2] in [3, 4]:  # RGB or RGBA
        gray_img = color.rgb2gray(image)
    elif image.ndim == 3 and image.shape[0] in [3, 4]:  # (C, H, W) 格式的彩色图像
        gray_img = color.rgb2gray(np.transpose(image, (1, 2, 0)))
    else:  # 已经是灰度图或者特殊的多通道图
        if image.ndim == 3:
            gray_img = np.squeeze(image, axis=-1) if image.shape[-1] == 1 else image[..., 0]
        else:
            gray_img = image

    # 归一化到0-255范围
    if gray_img.dtype != np.uint8:
        # 处理浮点型图像
        if gray_img.dtype.kind == "f":
            # 假设浮点型图像范围是[0, 1]
            normalized_img = (np.clip(gray_img, 0, 1) * 255).astype(np.uint8)
        else:
            # 处理其他整数类型
            normalized_img = (
                ((gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img)) * 255).astype(np.uint8)
                if np.max(gray_img) != np.min(gray_img)
                else np.zeros_like(gray_img, dtype=np.uint8)
            )
    else:
        normalized_img = gray_img

    return normalized_img


class CPImageDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.items, self.cls2idx, self.id2cls = scan_dataset(root_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label_idx, id_name = self.items[idx]
        try:
            image = io.imread(img_path)
            if self.transform:
                image = self.transform(image)
            return image, img_path, label_idx, id_name
        except Exception as e:
            logger.warning(f"skip bad image {img_path}: {e}")
            return np.zeros((256, 256, 3), dtype=np.uint8), img_path, label_idx, id_name


# ---------- 2. 推理 ---------- #
def _infer_one(model, images, diameter, **kwargs):
    """多张推理，返回 mask"""
    
    # Filter out custom keys that shouldn't go to Cellpose
    keys_to_remove = [
        "auto_target_enabled", "target_override", 
        "auto_target_quantile", "auto_target_threshold", 
        "auto_target_verbose", "ambiguous_policy"
    ]
    for k in keys_to_remove:
        kwargs.pop(k, None)

    # Cellpose 4.0.7 后支持批量处理, 但是只支持list(image)且image.shape=(H, W) 或 (H, W, 3)
    # 输出信息会有WARNING | Cannot stack images, processing one by one
    kwargs.setdefault("do_3D", False)
    kwargs.setdefault("channel_axis", None)
    kwargs.setdefault("batch_size", len(images))
    # Force single-channel config
    kwargs.setdefault("channels", [0, 0])

    result = model.eval(images, diameter=diameter, **kwargs)
    return result[0] if isinstance(result, (tuple, list)) else result


# ---------- 3. 分析 ---------- #
def _to_gray(image):
    """将图像转换为灰度图"""
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        from skimage import color

        # 确保输入图像是RGB或RGBA格式
        if image.shape[-1] in [3, 4]:  # RGB or RGBA
            image = color.rgb2gray(image)
            # 将浮点型灰度图像转换为与输入图像范围一致的格式
    return image


def _calculate_gray_value(red, green, blue):
    """根据RGB值计算灰度值，使用标准的加权平均法"""
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _analyze(image, mask, path, label_idx, id_name, classes):
    # 确保 mask 与 image  spatial 一致
    if mask.ndim == 3 and image.ndim == 3 and mask.shape != image.shape:
        # 常见：mask(Z,H,W)  vs  image(Z,H,W,3) → 取灰度
        if image.shape[-1] == 3:
            # image = image[..., 0]  # 简单用 R 通道
            image = _to_gray(image)

    # 保存原始图像用于RGB分析
    original_image = image

    # 采用v2版本的分析逻辑
    # 计算每个液滴的指标
    droplets_data = []
    # 对于灰度图像，直接使用image；对于彩色图像，需要创建一个灰度版本用于regionprops
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        gray_image = _to_gray(image)
        regions = measure.regionprops(mask, intensity_image=gray_image)
    else:
        regions = measure.regionprops(mask, intensity_image=image)

    regions = [r for r in regions if r.label > 0]  # 1.移除背景
    regions = [r for r in regions if r.area > 0 and r.intensity_image.sum() > 0]  # 2. 可选：去掉面积过小或强度=0 的空洞

    # 用于计算总体统计信息
    all_areas = []
    all_mean_grays = []
    all_total_grays = []

    # RGB统计信息
    all_total_reds = []
    all_total_greens = []
    all_total_blues = []

    all_mean_reds = []
    all_mean_greens = []
    all_mean_blues = []

    for i, region in enumerate(regions):
        # 基础几何信息
        area = region.area  # 液滴区域内像素点面积
        all_areas.append(area)

        # 如果原始图像是彩色图像，则计算RGB通道的统计信息
        if original_image.ndim == 3 and original_image.shape[-1] in [3, 4]:
            # 获取区域内的像素坐标
            region_coords = region.coords

            # 提取RGB值
            red_values = []
            green_values = []
            blue_values = []

            for coord in region_coords:
                y, x = coord
                red_values.append(original_image[y, x, 0])
                green_values.append(original_image[y, x, 1])
                blue_values.append(original_image[y, x, 2])

            total_red = np.sum(red_values)
            total_green = np.sum(green_values)
            total_blue = np.sum(blue_values)

            # 根据RGB值计算灰度值
            total_gray = np.sum(
                [_calculate_gray_value(r, g, b) for r, g, b in zip(red_values, green_values, blue_values)]
            )

            mean_red = total_red / area if area > 0 else 0
            mean_green = total_green / area if area > 0 else 0
            mean_blue = total_blue / area if area > 0 else 0
            mean_gray = total_gray / area if area > 0 else 0

            all_total_reds.append(total_red)
            all_total_greens.append(total_green)
            all_total_blues.append(total_blue)
            all_total_grays.append(total_gray)

            all_mean_reds.append(mean_red)
            all_mean_greens.append(mean_green)
            all_mean_blues.append(mean_blue)
            all_mean_grays.append(mean_gray)

            droplets_data.append(
                {
                    "droplet_id": i + 1,
                    "area_pixels": area,
                    "total_Gray": round(float(total_gray), 2),
                    "mean_Gray": round(float(mean_gray), 2),
                    "total_Red": round(float(total_red), 2),
                    "mean_Red": round(float(mean_red), 2),
                    "total_Green": round(float(total_green), 2),
                    "mean_Green": round(float(mean_green), 2),
                    "total_Blue": round(float(total_blue), 2),
                    "mean_Blue": round(float(mean_blue), 2),
                }
            )
        else:
            # 对于灰度图像
            total_gray = np.sum(region.intensity_image)  # 像素点灰度总和
            mean_gray = total_gray / area if area > 0 else 0  # 像素点灰度平均值

            all_total_grays.append(total_gray)
            all_mean_grays.append(mean_gray)

            # 对于灰度图像，RGB值与灰度值相同
            all_total_reds.append(total_gray)
            all_total_greens.append(total_gray)
            all_total_blues.append(total_gray)

            all_mean_reds.append(mean_gray)
            all_mean_greens.append(mean_gray)
            all_mean_blues.append(mean_gray)

            droplets_data.append(
                {
                    "droplet_id": i + 1,
                    "area_pixels": area,
                    "total_Gray": round(float(total_gray), 2),
                    "mean_Gray": round(float(mean_gray), 2),
                    "total_Red": round(float(total_gray), 2),
                    "mean_Red": round(float(mean_gray), 2),
                    "total_Green": round(float(total_gray), 2),
                    "mean_Green": round(float(mean_gray), 2),
                    "total_Blue": round(float(total_gray), 2),
                    "mean_Blue": round(float(mean_gray), 2),
                }
            )

    results_df = pd.DataFrame(droplets_data)

    # 计算总体统计信息
    total_area_pixels = sum(all_areas) if all_areas else 0
    total_gray = sum(all_total_grays) if all_total_grays else 0
    mean_gray = total_gray / total_area_pixels if total_area_pixels > 0 else 0

    total_red = sum(all_total_reds) if all_total_reds else 0
    total_green = sum(all_total_greens) if all_total_greens else 0
    total_blue = sum(all_total_blues) if all_total_blues else 0

    # 总体mean计算逻辑：对应的total除以像素点数
    mean_red = total_red / total_area_pixels if total_area_pixels > 0 else 0
    mean_green = total_green / total_area_pixels if total_area_pixels > 0 else 0
    mean_blue = total_blue / total_area_pixels if total_area_pixels > 0 else 0

    stats = {
        "total_droplets": len(results_df),
        "droplet_mean_area_pixels": (round(float(results_df["area_pixels"].mean()), 2) if not results_df.empty else 0),
        # 基于液滴的total与mean(命名前缀为droplet_total_, droplet_mean_)
        "droplet_mean_Gray": (round(float(results_df["mean_Gray"].mean()), 2) if not results_df.empty else 0),
        "droplet_mean_Red": (round(float(results_df["mean_Red"].mean()), 2) if not results_df.empty else 0),
        "droplet_mean_Green": (round(float(results_df["mean_Green"].mean()), 2) if not results_df.empty else 0),
        "droplet_mean_Blue": (round(float(results_df["mean_Blue"].mean()), 2) if not results_df.empty else 0),
        # 基于整体像素的total与mean(命名前缀为total_, mean_)
        "total_area_pixels": total_area_pixels,
        "total_Gray": round(float(total_gray), 2),
        "mean_Gray": (round(float(total_gray / total_area_pixels), 2) if total_area_pixels > 0 else 0),
        "total_Red": round(float(total_red), 2),
        "mean_Red": round(float(mean_red), 2),
        "total_Green": round(float(total_green), 2),
        "mean_Green": round(float(mean_green), 2),
        "total_Blue": round(float(total_blue), 2),
        "mean_Blue": round(float(mean_blue), 2),
        "image": os.path.basename(path),
        "label": classes[label_idx],
        "id": id_name,
    }

    return results_df, stats


# ---------- 4. 写盘 ---------- #
def _save_mask(image, masks, out_mask_file):
    """
    在原始图像上绘制mask轮廓并保存
    image  : (H, W) or (H, W, 3)  原图
    masks  : (H, W)               实例标签
    out_mask_file : 输出 *_mask.tif 路径
    """
    # 统一成 float 0-1
    if image.ndim == 2:
        img_rgb = np.stack([image] * 3, axis=-1)
    else:
        img_rgb = image.astype(np.float32)
    if img_rgb.max() > 1:
        img_rgb /= img_rgb.max()

    # 画轮廓
    outlined = mark_boundaries(img_rgb, masks, color=[0, 1, 0], mode="inner", background_label=0)

    # 在轮廓图像上添加编号
    outlined_vis = (outlined * 65535).astype(np.uint16)

    # 获取所有区域的中心点并添加编号
    # 传入intensity_image参数以确保可以访问region.intensity_image属性
    regions = measure.regionprops(masks, intensity_image=image)
    for region in regions:
        # 只对有效区域进行编号（跳过背景和无效区域）
        if region.label > 0 and region.area > 0 and region.intensity_image.sum() > 0:
            # 获取区域中心点坐标
            centroid = region.centroid
            y, x = int(centroid[0]), int(centroid[1])

            # 在图像上添加编号
            cv2.putText(
                outlined_vis,
                str(region.label),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 65535, 0),  # 绿色编号（BGR格式）
                1,
                cv2.LINE_AA,
            )

    # 0-255 uint16 tif
    io.imsave(out_mask_file, outlined_vis)
    logger.info(f"Outlined mask saved → {out_mask_file}")


def _save_csv(df, out_csv_file):
    df.to_csv(out_csv_file, index=False)


# ---------- 5. 批推理（OOM 自动减半） ---------- #
@torch.no_grad()
def _dynamic_batch_infer(
    model, original_images, processed_images, paths, labels, ids, diameter, out_dir, classes, save_mask=True, **kwargs
):
    batch_size = len(original_images)
    os.makedirs(out_dir, exist_ok=True)
    while batch_size >= 1:
        try:
            # 尝试批量处理
            if batch_size > 1:
                # 尝试堆叠图像进行批量处理
                try:
                    stacked_images = processed_images[:batch_size]
                    logger.info(f"Processing batch of {batch_size} images with shape {stacked_images[0].shape}")
                    masks = _infer_one(model, stacked_images, diameter, **kwargs)
                    if not isinstance(masks, list):
                        masks = [masks[i] for i in range(masks.shape[0])]  # 拆分批次结果
                except ValueError as e:
                    # 如果无法堆叠（图像尺寸不一致），则逐张处理
                    logger.warning(f"Cannot stack images, processing one by one: {e}")
                    masks = [_infer_one(model, img, diameter, **kwargs) for img in processed_images[:batch_size]]
            else:
                # 单张图像处理
                masks = [_infer_one(model, processed_images[0], diameter, **kwargs)]
                masks = [masks] if not isinstance(masks, list) else masks

            # 处理结果
            for orig_img, proc_img, mask, p, l, i in zip(
                original_images[:batch_size],
                processed_images[:batch_size],
                masks,
                paths[:batch_size],
                labels[:batch_size],
                ids[:batch_size],
            ):
                df, st = _analyze(orig_img, mask, p, l, i, classes)
                base = os.path.splitext(os.path.basename(p))[0]
                file_prefix = f"{classes[l]}_{i}"
                out_csv_file = os.path.join(out_dir, f"{file_prefix}_{base}.csv")

                _save_csv(df, out_csv_file)

                # 根据save_mask参数决定是否保存mask图像
                if save_mask:
                    out_mask_file = os.path.join(out_dir, f"{file_prefix}_{base}_mask.tif")
                    _save_mask(orig_img, mask, out_mask_file)

                yield st

            # 处理剩余图像
            for st in _dynamic_batch_infer(
                model,
                original_images[batch_size:],
                processed_images[batch_size:],
                paths[batch_size:],
                labels[batch_size:],
                ids[batch_size:],
                diameter,
                out_dir,
                classes,
                save_mask=save_mask,
                **kwargs,
            ):
                yield st
            return
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                batch_size //= 2
                torch.cuda.empty_cache()
                logger.warning(f"OOM → reduce batch_size to {batch_size}")
            else:
                raise


# ---------- 6. 统一入口 ---------- #
def batch_process_droplets(
    image_dir,
    output_dir,
    model="cyto3",
    diameter=None,
    mode="batch",
    batch_size=16,
    gpu=True, # Deprecated in favor of auto-detect, kept for signature compat
    num_workers=4,
    resample=None,
    max_parallel_batches=2,
    save_mask=True,
    calib_slope=1.0,
    calib_intercept=0.0,
    calib_direction="gray_to_log10",
    **kwargs,
):
    # ---------- 修正 1：正确解析 ds.items 三元组 ----------
    if mode == "single":
        if os.path.isfile(image_dir):
            im_paths = [image_dir]
            labels = [0]
            ids = ["default"]
            classes = ["default"]
        elif os.path.isdir(image_dir):
            im_paths = [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ]
            labels = [0] * len(im_paths)
            ids = ["default"] * len(im_paths)
            classes = ["default"] * len(im_paths)
        else:
            # 处理既不是文件也不是有效目录的情况
            logger.error(f"Invalid image_dir: {image_dir}")
            return pd.DataFrame()
    else:
        ds = CPImageDataset(image_dir)  # 三元组 (path, label_idx, id_name)
        classes = list(ds.cls2idx.keys())  # 为了 idx→字符串
        # 一次性建好只读列表，不再变动
        im_paths = [it[0] for it in ds.items]
        labels = [it[1] for it in ds.items]
        ids = [it[2] for it in ds.items]

    if not im_paths:
        logger.error("No images found!")
        return pd.DataFrame()

    pbar = tqdm(total=len(im_paths), desc="Processed")
    all_stats = []

    def _load(idx):
        try:
            original_img = io.imread(im_paths[idx])
            processed_img = normalize_image_to_uint8(original_img)  # 转换为灰度图并归一化
            return original_img, processed_img, im_paths[idx], labels[idx], ids[idx]
        except Exception as e:
            logger.warning(f"skip {im_paths[idx]}: {e}")
            return None, None, im_paths[idx], labels[idx], ids[idx]

    def process_batch(
        batch_original_images,
        batch_processed_images,
        batch_paths,
        batch_labels,
        batch_ids,
    ):
        """处理单个批次的函数"""
        batch_stats = []
        try:
            # --- Auto Target Detection & Policy Check ---
            # 1. Detect targets for all images in batch
            # 2. Filter images based on policy (error/skip/default)
            
            filtered_indices = [] # Indices of images to actually process with Cellpose
            skipped_results = []  # Placeholder results for skipped images
            
            batch_auto_info = [] # Store auto info for ALL images (index aligned with input batch)
            
            ambiguous_policy = kwargs.get("ambiguous_policy", "error")
            auto_enabled = kwargs.get("auto_target_enabled", False)
            
            for i, img in enumerate(batch_original_images):
                info = {
                    "target": kwargs.get("target_override", "unknown"), 
                    "score": None, "R_top": None, "G_top": None, 
                    "source": "user",
                    "ambiguous_reason": None,
                    "quantile": None, "threshold": None
                }
                
                proceed = True
                
                if auto_enabled:
                    q = kwargs.get("auto_target_quantile", 0.95)
                    thr = kwargs.get("auto_target_threshold", 0.10)
                    tgt, sc, rt, gt = infer_target_from_image(img, q=q, thr=thr)
                    
                    info["score"] = sc
                    info["R_top"] = rt
                    info["G_top"] = gt
                    info["source"] = "auto"
                    info["quantile"] = q
                    info["threshold"] = thr
                    
                    if tgt == "ambiguous":
                        info["ambiguous_reason"] = f"score={sc:.3f} abs<{thr}" if sc is not None else "format error"
                        if ambiguous_policy == "error":
                            err_msg = f"Auto-target ambiguous for {os.path.basename(batch_paths[i])} (score={sc}). Please specify --target miR-21 or miR-92a, or set --ambiguous_policy skip/default21/default92a."
                            logger.error(err_msg)
                            raise ValueError(err_msg)
                            
                        elif ambiguous_policy == "skip":
                            info["target"] = "ambiguous"
                            info["source"] = "auto"
                            logger.warning(f"Ambiguous target for {os.path.basename(batch_paths[i])}. Policy=skip.")
                            proceed = False
                            # Add to skipped_results for summary
                            skipped_results.append({
                                "image": os.path.basename(batch_paths[i]),
                                "label": classes[batch_labels[i]],
                                "id": batch_ids[i],
                                "target_detected": "ambiguous",
                                "target_source": "auto",
                                "auto_target_score": sc,
                                "auto_R_top": rt,
                                "auto_G_top": gt,
                                "auto_quantile": q,
                                "auto_threshold": thr,
                                "ambiguous_reason": info["ambiguous_reason"],
                                "mean_Gray": None,
                                "log10_concentration": None,
                                "concentration_M": None,
                                "concentration_fM": None,
                                "calib_slope_used": None,
                                "calib_intercept_used": None,
                                "total_droplets": 0
                            })
                            
                        elif ambiguous_policy == "default21":
                            info["target"] = "miR-21"
                            info["source"] = "auto_default"
                            logger.info(f"Ambiguous target for {os.path.basename(batch_paths[i])}. Policy=default21 -> miR-21")
                            proceed = True
                            
                        elif ambiguous_policy == "default92a":
                            info["target"] = "miR-92a"
                            info["source"] = "auto_default"
                            logger.info(f"Ambiguous target for {os.path.basename(batch_paths[i])}. Policy=default92a -> miR-92a")
                            proceed = True
                            
                    else:
                        info["target"] = tgt
                        proceed = True
                
                batch_auto_info.append(info)
                if proceed:
                    filtered_indices.append(i)
            
            # If no images left to process, just return skipped results (if any)
            if not filtered_indices:
                return skipped_results
            
            # Construct filtered batch
            f_orig = [batch_original_images[i] for i in filtered_indices]
            f_proc = [batch_processed_images[i] for i in filtered_indices]
            f_path = [batch_paths[i] for i in filtered_indices]
            f_lbls = [batch_labels[i] for i in filtered_indices]
            f_ids  = [batch_ids[i] for i in filtered_indices]
            
            # Run inference on filtered batch
            gen = _dynamic_batch_infer(
                cp_model,
                f_orig,
                f_proc,
                f_path,
                f_lbls,
                f_ids,
                diameter,
                output_dir,
                classes,
                save_mask=save_mask,
                **kwargs,
            )
            
            processed_count = 0
            for st in gen:
                # Map back to original index to get auto info
                original_idx = filtered_indices[processed_count]
                auto_info = batch_auto_info[original_idx]
                processed_count += 1
                
                # Determine slope/intercept
                current_slope = calib_slope
                current_intercept = calib_intercept
                current_direction = calib_direction
                
                tgt = auto_info.get("target")
                
                if auto_enabled:
                    if tgt == "miR-21":
                        current_slope = -2.57164
                        current_intercept = 43.32702
                        current_direction = "log10_to_gray"
                    elif tgt == "miR-92a":
                        current_slope = -2.11009
                        current_intercept = 44.32428
                        current_direction = "log10_to_gray"
                
                # ----- Concentration Calibration Logic -----
                gray_val = st.get("mean_Gray", 0.0)
                calib_x = None
                log10_conc = None
                conc_M = None
                conc_fM = None
                
                try:
                    if current_direction == "log10_to_gray":
                        if current_slope != 0:
                            calib_x = (gray_val - current_intercept) / current_slope
                            log10_conc = -calib_x
                        else:
                            calib_x = 0.0
                            log10_conc = 0.0
                            
                    elif current_direction == "gray_to_log10":
                        calib_x = current_slope * gray_val + current_intercept
                        log10_conc = calib_x
                    
                    if log10_conc is not None:
                        conc_M = 10 ** log10_conc
                        conc_fM = conc_M * 1e15
                except Exception as e:
                    logger.warning(f"Concentration calc failed: {e}")
                
                # Add to stats dict
                st["gray_value_used_for_calibration"] = float(gray_val)
                st["calib_x"] = float(calib_x) if calib_x is not None else None
                st["log10_concentration"] = float(log10_conc) if log10_conc is not None else None
                st["concentration_M"] = float(conc_M) if conc_M is not None else None
                st["concentration_fM"] = float(conc_fM) if conc_fM is not None else None
                
                # Add Audit Fields
                st["target_detected"] = tgt
                st["target_source"] = auto_info.get("source")
                st["auto_target_score"] = float(auto_info.get("score")) if auto_info.get("score") is not None else None
                st["auto_R_top"] = float(auto_info.get("R_top")) if auto_info.get("R_top") is not None else None
                st["auto_G_top"] = float(auto_info.get("G_top")) if auto_info.get("G_top") is not None else None
                st["auto_quantile"] = auto_info.get("quantile")
                st["auto_threshold"] = auto_info.get("threshold")
                st["ambiguous_reason"] = auto_info.get("ambiguous_reason")
                st["calib_slope_used"] = float(current_slope)
                st["calib_intercept_used"] = float(current_intercept)
                
                batch_stats.append(st)
                
            # Merge skipped results
            batch_stats.extend(skipped_results)
            
        except Exception as e:
            # Re-raise if it's the specific ambiguous policy error
            if "Auto-target ambiguous" in str(e) and "ambiguous_policy" in str(e):
                raise e
                
            logger.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
        return batch_stats

    logger.info("Loading Cellpose model …")
    # cp_model = models.CellposeModel(gpu=gpu, pretrained_model=model)
    cp_model = load_cellpose_model_robust(model_name=model)

    with ThreadPoolExecutor(max_workers=num_workers) as load_executor:
        futures = {load_executor.submit(_load, i): i for i in range(len(im_paths))}

        # 收集图像批次以进行批处理
        (
            original_images_batch,
            processed_images_batch,
            paths_batch,
            labels_batch,
            ids_batch,
        ) = ([], [], [], [], [])
        batch_futures = []

        # 使用线程池并行处理批次
        with ThreadPoolExecutor(max_workers=max_parallel_batches) as batch_executor:
            for fut in as_completed(futures):
                original_img, processed_img, p, l, i = fut.result()
                if original_img is None:
                    pbar.update(1)
                    continue

                # 收集图像到批次中
                original_images_batch.append(original_img)
                processed_images_batch.append(processed_img)
                paths_batch.append(p)
                labels_batch.append(l)
                ids_batch.append(i)

                # 当批次达到指定大小时提交处理任务
                if len(processed_images_batch) >= batch_size:
                    # 提交批次处理任务
                    batch_future = batch_executor.submit(
                        process_batch,
                        original_images_batch.copy(),
                        processed_images_batch.copy(),
                        paths_batch.copy(),
                        labels_batch.copy(),
                        ids_batch.copy(),
                    )
                    batch_futures.append(batch_future)

                    # 重置批次
                    (
                        original_images_batch,
                        processed_images_batch,
                        paths_batch,
                        labels_batch,
                        ids_batch,
                    ) = ([], [], [], [], [])

            # 处理剩余的图像
            if processed_images_batch:
                batch_future = batch_executor.submit(
                    process_batch,
                    original_images_batch,
                    processed_images_batch,
                    paths_batch,
                    labels_batch,
                    ids_batch,
                )
                batch_futures.append(batch_future)

            # 等待所有批次处理完成并收集结果
            for batch_future in as_completed(batch_futures):
                batch_stats = batch_future.result()
                for st in batch_stats:
                    all_stats.append(st)
                    pbar.update(1)

    pbar.close()

    if all_stats:
        summary = pd.DataFrame(all_stats)
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "batch_summary_statistics.csv")
        
        # Ensure new columns are in the output and maybe reorder for clarity
        cols = summary.columns.tolist()
        priority_cols = [
            "image", "label", "id", 
            "target_detected", "target_source", "ambiguous_reason",
            "auto_target_score", "auto_R_top", "auto_G_top",
            "gray_value_used_for_calibration", "log10_concentration", 
            "concentration_M", "concentration_fM", 
            "total_droplets", "mean_Gray"
        ]
        # Sort cols: priority first, then rest
        new_cols = []
        for c in priority_cols:
            if c in cols:
                new_cols.append(c)
        for c in cols:
            if c not in new_cols:
                new_cols.append(c)
        
        summary = summary[new_cols]
        
        summary.to_csv(summary_path, index=False)
        logger.info(f"All done! Summary → {summary_path}")
        return summary
    else:
        logger.warning("No successful image.")
        return pd.DataFrame()



def extract_droplet_features_from_array(img: np.ndarray, model_name="cyto3", max_side=1024) -> pd.DataFrame:
    """
    参数
    ----
    img : np.ndarray
        H x W x 3 的 uint8 彩色图像（Gradio Image 上传后传进来的数组）。

    返回
    ----
    df : pandas.DataFrame
        包含每个液滴特征的DataFrame，列名为：
        ['gray_mean', 'r_mean', 'g_mean', 'b_mean']
    """
    logger.info(f"Extracting features from image with shape: {img.shape}, dtype: {img.dtype}")
    
    # Ensure image is uint8
    if img.dtype != np.uint8:
        logger.warning(f"Image dtype is {img.dtype}, normalizing to uint8")
        img = normalize_image_to_uint8(img)

    # 1. 初始化模型 (使用 cyto3 模型，自动选择设备)
    # Note: Function signature updated to accept model_name, default 'cyto3'
    model = load_cellpose_model_robust(model_name=model_name)

    # 2. 推理 (分割)
    logger.info(f"start: Running segmentation for image (H,W)={img.shape[:2]}")
    t0 = time.time()
    try:
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > int(max_side):
            scale = float(max_side) / float(max(h, w))
        seg_input = img
        if seg_input.ndim == 3 and seg_input.shape[-1] >= 3:
            seg_input = _to_gray(seg_input)
        if scale < 1.0:
            from skimage.transform import resize
            seg_input = resize(seg_input, (int(h * scale), int(w * scale)), order=1, preserve_range=True, anti_aliasing=True).astype(seg_input.dtype)
        mask = _infer_one(model, [seg_input], diameter=None)
        if isinstance(mask, (list, tuple)):
            mask = mask[0] if len(mask) > 0 else None
        if scale < 1.0 and mask is not None:
            from skimage.transform import resize
            mask = resize(mask, (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    t1 = time.time()
    
    if mask is None:
        logger.error("Inference returned None mask.")
        return pd.DataFrame()

    unique_labels = np.unique(mask)
    n_labels = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
    logger.info(f"end: Segmentation done in {t1-t0:.2f} s, labels={n_labels}")

    # 3. 分析 (特征提取)
    # DEBUG: Check mask type and shape
    print(
        "[DEBUG] extract_droplet_features_from_array: type(mask) =",
        type(mask),
        "has_ndim =", hasattr(mask, "ndim"),
        "shape =", getattr(mask, "shape", None)
    )

    # Normalize mask if it is a list/tuple
    if isinstance(mask, (list, tuple)):
        print(f"[DEBUG] mask is list/tuple with length {len(mask)}")
        if len(mask) > 0:
            mask = mask[0] # Assume the first element is the mask
        else:
            logger.error("[ERROR] mask list/tuple is empty")
            return pd.DataFrame()
        
        print(
            "[DEBUG] after normalization, type(mask) =",
            type(mask),
            "ndim =", getattr(mask, "ndim", None),
            "shape =", getattr(mask, "shape", None)
        )

    # 确保 mask 与 image spatial 一致
    if mask.ndim == 3 and img.ndim == 3 and mask.shape != img.shape:
        if img.shape[-1] == 3:
            img_gray = _to_gray(img)
        else:
            img_gray = img
    else:
        img_gray = img
    
    # Ensure img_gray is suitable for regionprops
    if img_gray.ndim == 3 and img_gray.shape[-1] == 3:
         img_gray = _to_gray(img_gray)

    try:
        regions = measure.regionprops(mask, intensity_image=img_gray)
    except Exception as e:
        logger.error(f"regionprops failed: {e}")
        return pd.DataFrame()

    regions = [r for r in regions if r.label > 0]
    regions = [r for r in regions if r.area > 0]
    
    logger.info(f"Found {len(regions)} valid regions.")

    features = []
    for r in regions:
        # 提取特征
        # 注意：regionprops 的 intensity_image 是灰度图
        # 我们需要回到原始 RGB 图像提取颜色特征
        
        # 获取 bounding box
        minr, minc, maxr, maxc = r.bbox
        
        # 提取该区域的 mask
        # r.image 是该区域的二值 mask (在 bbox 内)
        region_mask = r.image
        
        # 提取对应的 RGB 像素
        # img[minr:maxr, minc:maxc] 是 bbox 内的图像
        # 我们只取 mask 为 True 的部分
        if img.ndim == 3:
            crop = img[minr:maxr, minc:maxc]
            # region_mask 是 (H, W)，crop 是 (H, W, 3)
            # 扩展 mask 维度
            try:
                pixels = crop[region_mask] # (N, 3)
            except IndexError as e:
                logger.warning(f"IndexError extracting pixels for region {r.label}: {e}")
                continue
            
            if len(pixels) == 0:
                continue
                
            r_mean = pixels[:, 0].mean()
            g_mean = pixels[:, 1].mean()
            b_mean = pixels[:, 2].mean()
            gray_mean = _calculate_gray_value(r_mean, g_mean, b_mean)
        else:
            # 灰度图情况
            crop = img[minr:maxr, minc:maxc]
            try:
                pixels = crop[region_mask]
            except IndexError as e:
                logger.warning(f"IndexError extracting pixels for region {r.label}: {e}")
                continue
                
            if len(pixels) == 0:
                continue
            gray_mean = pixels.mean()
            r_mean = gray_mean
            g_mean = gray_mean
            b_mean = gray_mean

        features.append({
            "gray_mean": gray_mean,
            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,
            "area": r.area
        })

    df = pd.DataFrame(features)
    logger.info(f"Extracted features for {len(df)} droplets.")
    return df

def compute_extended_features_v2(df: pd.DataFrame) -> np.ndarray:
    """
    Computes the 46-dimensional feature vector required by the miR-92a SVM model.
    Encapsulates all statistical calculations and interactions.
    
    Returns:
        np.ndarray: shape (1, 46)
    """
    from scipy import stats
    
    # Map logical channel names to DataFrame columns
    channels = {
        'Gray': 'gray_mean',
        'Red': 'r_mean',
        'Green': 'g_mean',
        'Blue': 'b_mean'
    }
    
    # 1. Calculate Statistics for each channel
    stats_data = {}
    for ch_name, col in channels.items():
        if col not in df.columns:
            # Fallback if column missing
            series = pd.Series([0])
        else:
            series = df[col]
        
        # Base mean
        stats_data[f"mean_{ch_name}"] = series.mean()
        
        # Droplet means (same as mean, but naming consistency for model)
        stats_data[f"droplet_mean_{ch_name}"] = series.mean()

        # Extended stats
        if len(series) > 1:
            stats_data[f"std_{ch_name}"] = series.std(ddof=1)
            stats_data[f"skew_{ch_name}"] = stats.skew(series)
        else:
            stats_data[f"std_{ch_name}"] = 0.0
            stats_data[f"skew_{ch_name}"] = 0.0
            
        stats_data[f"max_{ch_name}"] = series.max()
        stats_data[f"min_{ch_name}"] = series.min()
        stats_data[f"median_{ch_name}"] = series.median()

    # 2. Construct Feature List in Correct Order
    stat_order = ['std', 'max', 'min', 'median', 'skew']
    channel_order = ['Gray', 'Red', 'Green', 'Blue']
    
    feature_list = []
    
    # A. Droplet Means (4)
    for ch in channel_order:
        feature_list.append(stats_data.get(f"droplet_mean_{ch}", 0.0))
        
    # B. Global Means (4)
    for ch in channel_order:
        feature_list.append(stats_data.get(f"mean_{ch}", 0.0))
        
    # C. Detailed Stats (5 * 4 = 20)
    for ch in channel_order:
        for stat in stat_order:
            feature_list.append(stats_data.get(f"{stat}_{ch}", 0.0))
            
    # D. Interactions (3 * 6 = 18)
    base_cols = ['mean_Gray', 'mean_Red', 'mean_Green', 'mean_Blue']
    
    for i, col1 in enumerate(base_cols):
        for j, col2 in enumerate(base_cols):
            if i < j:
                val1 = stats_data.get(col1, 0.0)
                val2 = stats_data.get(col2, 0.0)
                
                # Ratio
                feature_list.append(val1 / (val2 + 1e-8))
                # Diff
                feature_list.append(val1 - val2)
                # Product
                feature_list.append(val1 * val2)
                
    return np.array(feature_list).reshape(1, -1)

def compute_features_mir21(df: pd.DataFrame) -> np.ndarray:
    """
    Computes the 10-dimensional feature vector required by the miR-21 SVM model.
    Encapsulates statistical calculations.
    
    Features:
    ['total_droplets', 'droplet_mean_Gray', 'droplet_mean_Red', 'droplet_mean_Green', 'droplet_mean_Blue', 
     'total_area_pixels', 'mean_Gray', 'mean_Red', 'mean_Green', 'mean_Blue']
    
    Returns:
        np.ndarray: shape (1, 10)
    """
    total_droplets = len(df)
    
    # Droplet-level means (average of means)
    droplet_mean_Gray = df['gray_mean'].mean()
    droplet_mean_Red = df['r_mean'].mean()
    droplet_mean_Green = df['g_mean'].mean()
    droplet_mean_Blue = df['b_mean'].mean()
    
    # Global stats (weighted by area)
    total_area_pixels = df['area'].sum() if 'area' in df.columns else 0
    
    if total_area_pixels > 0 and 'area' in df.columns:
        # Calculate global mean by weighting droplet means by their area
        # (assuming droplet mean is uniform within droplet)
        mean_Gray = (df['gray_mean'] * df['area']).sum() / total_area_pixels
        mean_Red = (df['r_mean'] * df['area']).sum() / total_area_pixels
        mean_Green = (df['g_mean'] * df['area']).sum() / total_area_pixels
        mean_Blue = (df['b_mean'] * df['area']).sum() / total_area_pixels
    else:
        mean_Gray = droplet_mean_Gray
        mean_Red = droplet_mean_Red
        mean_Green = droplet_mean_Green
        mean_Blue = droplet_mean_Blue

    features = np.array([
        total_droplets,
        droplet_mean_Gray,
        droplet_mean_Red,
        droplet_mean_Green,
        droplet_mean_Blue,
        total_area_pixels,
        mean_Gray,
        mean_Red,
        mean_Green,
        mean_Blue
    ])
    
    return np.nan_to_num(features).reshape(1, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Droplet Analysis with Cellpose v3/v4")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory or single image file")
    parser.add_argument("--output_dir", type=str, default="./outputs/cellpose_results", help="Directory to save results")
    parser.add_argument("--model", type=str, default="cyto3", help="Cellpose model name (e.g., cyto3, cpsam)")
    parser.add_argument("--diameter", type=float, default=None, help="Cell diameter (optional)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for loading images")
    parser.add_argument("--max_parallel_batches", type=int, default=2, help="Max parallel batches")
    parser.add_argument("--save_mask", action="store_true", default=True, help="Save mask images")
    parser.add_argument("--no_save_mask", action="store_false", dest="save_mask", help="Do not save mask images")
    
    # Calibration args
    parser.add_argument("--target", type=str, default="auto", choices=["auto", "miR-21", "miR-92a"], help="Target miRNA")
    parser.add_argument("--calib_mode", type=str, default="builtin", choices=["builtin", "custom"], help="Calibration mode")
    parser.add_argument("--calib_slope", type=float, default=None, help="Calibration slope (a)")
    parser.add_argument("--calib_intercept", type=float, default=None, help="Calibration intercept (b)")
    parser.add_argument("--calib_direction", type=str, default="log10_to_gray", choices=["gray_to_log10", "log10_to_gray"], 
                        help="Calibration direction: gray_to_log10 (log10C = a*Gray + b) or log10_to_gray (Gray = a*log10C + b)")

    # Auto target args
    parser.add_argument("--auto_target_quantile", type=float, default=0.95, help="Quantile for top pixels")
    parser.add_argument("--auto_target_threshold", type=float, default=0.10, help="Score threshold for auto detection")
    parser.add_argument("--auto_target_verbose", action="store_true", help="Print auto detection details")
    parser.add_argument("--ambiguous_policy", type=str, default="error", 
                        choices=["error", "skip", "default21", "default92a"],
                        help="Policy for ambiguous targets in auto mode: error, skip, default21, default92a")

    args = parser.parse_args()

    # Resolve slope/intercept
    final_slope = args.calib_slope
    final_intercept = args.calib_intercept
    final_direction = args.calib_direction
    
    # Auto Target Flag
    auto_target_enabled = False
    
    if args.target == "auto":
        auto_target_enabled = True
        # Slope/Intercept will be resolved per image inside batch_process_droplets
        # We set default placeholders here to avoid NoneType errors if custom logic fails
        if final_slope is None: final_slope = 0.0
        if final_intercept is None: final_intercept = 0.0
    else:
        # Manual target: resolve globally as before
        if args.calib_mode == "builtin":
            if args.target == "miR-21":
                # y = -2.57164 * x + 43.32702 (y=Gray, x=-log10C)
                final_slope = -2.57164
                final_intercept = 43.32702
                final_direction = "log10_to_gray"
            elif args.target == "miR-92a":
                # y = -2.11009 * x + 44.32428
                final_slope = -2.11009
                final_intercept = 44.32428
                final_direction = "log10_to_gray"
        else:
            # Custom mode
            if final_slope is None: final_slope = 1.0
            if final_intercept is None: final_intercept = 0.0

    # Pass args to batch_process_droplets
    batch_process_droplets(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model=args.model,
        diameter=args.diameter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_parallel_batches=args.max_parallel_batches,
        save_mask=args.save_mask,
        calib_slope=final_slope,
        calib_intercept=final_intercept,
        calib_direction=final_direction,
        gpu=False, # Auto-detected inside now
        
        # New args for auto target
        auto_target_enabled=auto_target_enabled,
        auto_target_quantile=args.auto_target_quantile,
        auto_target_threshold=args.auto_target_threshold,
        auto_target_verbose=args.auto_target_verbose,
        target_override=args.target if args.target != "auto" else "unknown", # For logging user intent
        ambiguous_policy=args.ambiguous_policy
    )
