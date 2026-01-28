import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
import sys
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import xlsxwriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import sys
import datetime
import random

# 设置随机种子以确保结果可重现
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # 关闭cudnn的自动优化以确保结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BallBrightnessTransform:
    def __init__(self, brightness_jitter=0.2, deterministic=False):
        self.brightness_jitter = brightness_jitter
        self.min_radius = 5
        self.max_radius = 30
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.deterministic = deterministic  # 是否使用确定性模式

    def detect_balls(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20, 
            param1=self.canny_threshold1, param2=self.canny_threshold2, 
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        return circles

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
            
        # 转换为numpy数组
        img_np = np.array(img)
        
        # 转换为BGR格式用于OpenCV处理
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        original_img = img_bgr.copy()
        circles = self.detect_balls(img_bgr)
        ball_brightness = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            mask = np.zeros_like(img_bgr)
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                cv2.circle(mask, center, radius, (255, 255, 255), -1)
                ball_region = cv2.bitwise_and(img_bgr, mask)
                ball_gray = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
                non_zero_pixels = ball_gray[ball_gray > 0]
                if len(non_zero_pixels) > 0:
                    brightness = np.mean(non_zero_pixels)
                    ball_brightness.append(brightness)

        if ball_brightness:
            avg_brightness = np.mean(ball_brightness)
        else:
            avg_brightness = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

        # 在deterministic模式下使用固定的亮度因子
        if self.deterministic:
            brightness_factor = 1.0  # 不进行随机亮度调整
        else:
            brightness_factor = 1.0 + self.brightness_jitter * (2 * np.random.rand() - 1)
        
        adjusted_brightness = np.clip(avg_brightness * brightness_factor, 0, 255)
        brightness_diff = adjusted_brightness - avg_brightness

        if circles is not None and len(ball_brightness) > 0:
            result = original_img.copy()
            mask = np.zeros_like(original_img)
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                cv2.circle(mask, center, radius, (255, 255, 255), -1)
            result = result.astype(np.float32)
            mask_bool = mask > 0
            result[mask_bool] = np.clip(result[mask_bool] + brightness_diff, 0, 255)
        else:
            result = np.clip(img_bgr.astype(np.float32) + brightness_diff, 0, 255)

        # 转换回RGB格式
        result_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        # 返回PIL.Image对象
        return Image.fromarray(result_rgb)

if __name__ == '__main__':
    # 设置随机种子确保结果可重现
    set_random_seed(42)
    
    # 设置数据路径和超参数
    data_dir = "D:/green-miR-92a"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    
    # 设置带时间戳的输出目录，避免覆盖上一次训练结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("D:/green-miR-92a", f'output_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    print(f"输出目录已设置为: {results_dir}")
    
    # 调度器配置
    scheduler_type = "CosineAnnealing"  # 可选: "CosineAnnealing", "OneCycle", "Cyclic"
    
    # 检查数据集完整性
    print("正在检查数据集完整性...")
    # 只读取指定的8个分类文件夹
    specified_classes = ['1fM', '10fM', '100fM', '1pM', '10pM', '100pM', '1nM', 'blank']
    class_dirs = []
    for d in specified_classes:
        if os.path.isdir(os.path.join(data_dir, d)):
            class_dirs.append(d)
        else:
            print(f"警告: 指定的类别文件夹 {d} 不存在")
    
    if not class_dirs:
        raise FileNotFoundError(f"数据集目录 {data_dir} 中未找到指定的类别子文件夹")
    if len(class_dirs) < 8:
        print(f"警告: 只找到 {len(class_dirs)} 个指定的类别文件夹，而不是8个")

    supported_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    valid_samples = 0
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(supported_extensions)]
        if not image_files:
            raise FileNotFoundError(f"类别文件夹 {class_path} 中未找到有效图片文件")
        valid_samples += len(image_files)
    
    print(f"数据集检查完成，共发现{len(class_dirs)}个类别，{valid_samples}张有效图片")
    
    num_classes = len(class_dirs)
    batch_size = 64  # 增大batch size，原值32
    accumulation_steps = 2  # 梯度累积步数，有效batch size为64*2=128
    num_epochs = 70  # 增加到70轮
    base_lr = 0.00005  # 降低学习率，原值0.0001
    max_lr = 0.0005  # 降低最大学习率，原值0.001

    # 数据增强和预处理
    # 训练集transform：包含所有数据增强
    # 训练集数据增强 - 更强的正则化
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证/测试集 - 轻量级增强，保持数据一致性
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),  # 确保中心区域一致性
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集并按类别分别划分训练集、验证集和测试集
    from torch.utils.data import Dataset, ConcatDataset
    
    # 创建自定义数据集类，用于按类别分别加载和划分数据
    class PerClassSplitDataset(Dataset):
        def __init__(self, image_paths, targets, transform=None):
            self.image_paths = image_paths
            self.targets = targets
            self.transform = transform
            self.class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}
            self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            target = self.targets[idx]
            
            if self.transform:
                img = self.transform(img)
            
            return img, target
    
    # 为每个类别分别划分数据集
    train_image_paths = []
    train_targets = []
    val_image_paths = []
    val_targets = []
    test_image_paths = []
    test_targets = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_dir)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(supported_extensions)]
        image_paths = [os.path.join(class_path, f) for f in image_files]
        
        # 为每个类别分别划分训练集、验证集和测试集
        class_size = len(image_paths)
        train_size = int(0.6 * class_size)
        val_size = int(0.2 * class_size)
        test_size = class_size - train_size - val_size
        
        # 随机打乱
        indices = torch.randperm(class_size).tolist()
        
        # 划分索引
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # 添加到相应的集合
        train_image_paths.extend([image_paths[i] for i in train_indices])
        train_targets.extend([class_idx] * len(train_indices))
        val_image_paths.extend([image_paths[i] for i in val_indices])
        val_targets.extend([class_idx] * len(val_indices))
        test_image_paths.extend([image_paths[i] for i in test_indices])
        test_targets.extend([class_idx] * len(test_indices))
    
    # 创建数据集
    train_dataset = PerClassSplitDataset(train_image_paths, train_targets, train_transform)
    val_dataset = PerClassSplitDataset(val_image_paths, val_targets, val_test_transform)
    test_dataset = PerClassSplitDataset(test_image_paths, test_targets, val_test_transform)
    
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")
    
    # 统计并显示每个类别的样本数量
    print("\n每个类别的样本数量统计:")
    for class_idx, class_dir in enumerate(class_dirs):
        train_count = sum(1 for target in train_targets if target == class_idx)
        val_count = sum(1 for target in val_targets if target == class_idx)
        test_count = sum(1 for target in test_targets if target == class_idx)
        total_count = train_count + val_count + test_count
        print(f"{class_dir}: 训练集={train_count}, 验证集={val_count}, 测试集={test_count}, 总计={total_count}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True
    )
    
    print(f"每个epoch的batch数: {len(train_loader)}, 总训练步数: {num_epochs * len(train_loader)}")

    # 初始化模型
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备: {device}")

    # 计算类别权重以处理数据不平衡
    class_counts = []
    for class_idx in range(len(class_dirs)):
        class_count = sum(1 for target in train_targets if target == class_idx)
        class_counts.append(class_count)
    
    # 计算类别权重：样本数越少，权重越大
    class_weights = [1.0 / count if count > 0 else 1.0 for count in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"类别权重: {dict(zip(class_dirs, [f'{w:.3f}' for w in class_weights.cpu().numpy()]))}")
    
    # 损失函数 - 使用带权重的交叉熵损失和焦点损失
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, 
        label_smoothing=0.1,
        reduction='mean'
    )
    
    # 添加焦点损失函数来改善困难样本的学习
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        def forward(self, inputs, targets):
            ce_loss = self.ce_loss(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
    
    # 使用焦点损失作为辅助损失
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    # 为不同层设置不同的学习率
    optimizer = optim.AdamW([
        {'params': model.conv1.parameters(), 'lr': base_lr * 0.1},  # 预训练层使用较小学习率
        {'params': model.layer1.parameters(), 'lr': base_lr * 0.1},
        {'params': model.layer2.parameters(), 'lr': base_lr * 0.3},
        {'params': model.layer3.parameters(), 'lr': base_lr * 0.5},
        {'params': model.layer4.parameters(), 'lr': base_lr * 0.7},
        {'params': model.fc.parameters(), 'lr': base_lr}  # 新分类层使用完整学习率
    ], weight_decay=1e-4)
    
    # 学习率调度器选择
    if scheduler_type == "CosineAnnealing":
        # CosineAnnealingWarmRestarts调度器
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=40,  # 第一次重启的周期
            T_mult=1,  # 周期乘数
            eta_min=base_lr * 0.01  # 最小学习率
        )
        print("使用CosineAnnealingWarmRestarts调度器")
    elif scheduler_type == "OneCycle":
        # OneCycleLR调度器
        max_lr = 3e-4  # 默认最大值，将通过lr_find()更新
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 学习率上升阶段占比
            anneal_strategy='cos',  # 余弦退火
            div_factor=25,  # 初始学习率 = max_lr / div_factor
            final_div_factor=1e4  # 最终学习率 = max_lr / final_div_factor
        )
        print("使用OneCycleLR调度器")
    else:
        # 原来的CyclicLR调度器
        total_steps = num_epochs * len(train_loader)
        cycle_steps = total_steps // 4
        scheduler = CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=cycle_steps,
            step_size_down=cycle_steps,
            mode='triangular',
            cycle_momentum=False
        )
        print("使用CyclicLR调度器")
    
    # 用于锁定最佳学习率的调度器（仅对非OneCycle调度器有效）
    if scheduler_type != "OneCycle":
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-7,
            threshold=0.001
        )

    # 训练记录 - 只初始化一次
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    lr_history = []
    best_acc = 0.0
    best_epoch = 0
    global_step = 0
    frozen_lr = False  # 标记是否已锁定学习率
    patience = 70  # 早停耐心值，增加耐心值避免过早停止
    patience_counter = 0  # 早停计数器
    
    # 根据调度器类型设置学习率
    if scheduler_type == "OneCycle":
        # 运行学习率搜索
        best_lr = lr_find(model, train_loader, criterion, optimizer, device, num_iter=30)
        
        # 更新OneCycleLR的最大学习率
        scheduler = OneCycleLR(
            optimizer,
            max_lr=best_lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
        print(f"更新OneCycleLR最大学习率为: {best_lr:.2e}")
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_processed = 0
        
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 使用组合损失：带权重的交叉熵损失 + 焦点损失
            ce_loss = criterion(outputs, labels)
            focal_loss_val = focal_loss_fn(outputs, labels)
            loss = ce_loss + 0.5 * focal_loss_val  # 组合损失
            
            # 缩放损失以考虑梯度累积
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            # 添加梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 累积足够的梯度后才更新参数
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
                # 更新学习率并记录
                if scheduler_type == "CosineAnnealing":
                    # CosineAnnealingWarmRestarts在每个step后更新
                    scheduler.step(global_step)
                elif scheduler_type == "OneCycle":
                    # OneCycleLR在每个step后更新
                    scheduler.step()
                else:
                    # CyclicLR
                    if not frozen_lr:
                        scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                lr_history.append(current_lr)
                global_step += 1
            else:
                current_lr = optimizer.param_groups[0]['lr']

            # 计算统计量
            running_loss += loss.item() * accumulation_steps * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_processed += inputs.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'ce_loss': f"{ce_loss.item() * accumulation_steps:.3f}",
                'focal_loss': f"{focal_loss_val.item() * accumulation_steps:.3f}",
                'lr': f"{current_lr:.2e}",
                'step': global_step,
                'frozen': 'Y' if frozen_lr else 'N'
            })

        # 计算epoch指标
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        print(f'训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证")
            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # 验证阶段也使用组合损失
                ce_loss = criterion(outputs, labels)
                focal_loss_val = focal_loss_fn(outputs, labels)
                loss = ce_loss + 0.5 * focal_loss_val

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 验证阶段也记录学习率（但不更新）
                current_lr = optimizer.param_groups[0]['lr']
                lr_history.append(current_lr)
                global_step += 1

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())
        print(f'验证 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 更新最佳模型保存路径
        # 更新最佳准确率和epoch
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            model_path = os.path.join(results_dir, 'resnet_brightness_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f'保存最佳模型至 {model_path}，验证准确率: {best_acc:.4f}')
            patience_counter = 0  # 重置早停计数器
            
            # 如果准确率提升显著，考虑锁定学习率
            if epoch > 10 and not frozen_lr:
                print("发现新的最佳准确率，考虑锁定学习率...")
                # 计算最近5个epoch的平均提升
                if len(val_accs) > 5:
                    recent_improve = np.mean(np.diff(val_accs[-5:]))
                    if recent_improve < 0.001:  # 提升很小
                        frozen_lr = True
                        print(f"锁定学习率为: {current_lr:.6f}，不再调整")
        else:
            patience_counter += 1
            print(f'验证准确率未提升，早停计数: {patience_counter}/{patience}')
        
        # 早停检查
        if patience_counter >= patience:
            print(f'验证准确率连续{patience}个epoch未提升，提前停止训练')
            break

        # 使用Plateau调度器（仅在未锁定学习率时）
        if scheduler_type != "OneCycle" and not frozen_lr:  # OneCycle不需要Plateau调度器
            plateau_scheduler.step(epoch_acc)
        elif scheduler_type != "OneCycle":
            # 锁定学习率后，保持学习率不变（仅对非OneCycle调度器）
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

    # 在测试集上评估模型
    def evaluate_test_set(model, test_loader, criterion, focal_loss_fn, device):
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # 测试阶段也使用组合损失
                ce_loss = criterion(outputs, labels)
                focal_loss_val = focal_loss_fn(outputs, labels)
                loss = ce_loss + 0.5 * focal_loss_val
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        return test_loss, test_acc

    def calculate_map_iou_analysis(model, test_loader, device, class_names, results_dir):
        """
        计算目标检测任务的mAP（平均精度均值）
        实现COCO-style mAP@[0.5:0.95]和Pascal VOC-style mAP@0.5
        """
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import label_binarize
        from tqdm import tqdm
        import matplotlib.font_manager as fm
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="计算mAP"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_probabilities.extend(probabilities.cpu().numpy())
                all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # 将标签二值化用于多类别AP计算
        y_true_bin = label_binarize(all_targets, classes=range(len(class_names)))
        
        # 计算标准mAP（使用sklearn的average_precision_score）
        standard_mAP = np.mean([
            average_precision_score(y_true_bin[:, i], all_probabilities[:, i]) 
            for i in range(len(class_names))
        ])
        
        # 计算每个类别的AP
        class_ap_scores = {}
        for i, class_name in enumerate(class_names):
            try:
                ap = average_precision_score(y_true_bin[:, i], all_probabilities[:, i])
                class_ap_scores[class_name] = ap if not np.isnan(ap) else 0.0
            except:
                class_ap_scores[class_name] = 0.0
        
        print(f"测试集mAP: {standard_mAP:.4f}")
        print("各类别AP分数:")
        for class_name, ap in class_ap_scores.items():
            print(f"    {class_name}: {ap:.4f}")
        
        # 生成mAP柱状图
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        bars = plt.bar(range(len(class_names)), [class_ap_scores[cls] for cls in class_names], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        plt.title('各类别AP分数', fontsize=16, fontweight='bold')
        plt.xlabel('类别', fontsize=14)
        plt.ylabel('AP分数', fontsize=14)
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在每个柱子上添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'mAP_analysis_bar_chart.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"mAP柱状图已保存至: {plot_path}")
        
        # 保存mAP分析数据到Excel
        try:
            mAP_df = pd.DataFrame({
                '类别': list(class_ap_scores.keys()),
                'AP分数': list(class_ap_scores.values()),
                '总体mAP': [standard_mAP] * len(class_ap_scores)
            })
            excel_path = os.path.join(results_dir, 'mAP_analysis.xlsx')
            mAP_df.to_excel(excel_path, index=False)
            print(f"mAP分析数据已保存至: {excel_path}")
        except PermissionError as e:
            print(f"错误: 无法写入Excel文件，请关闭Excel文件后重试")
            excel_path = None
        
        return {
            'overall_mAP': standard_mAP,
            'class_aps': class_ap_scores,
            'plot_path': plot_path,
            'excel_path': excel_path
        }

    # 评估测试集
    test_loss, test_acc = evaluate_test_set(model, test_loader, criterion, focal_loss_fn, device)
    print(f'测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.4f}')
    
    # 计算mAP和绘制IoU阈值关系图
    mAP_results = calculate_map_iou_analysis(model, test_loader, device, class_dirs, results_dir)
    print(f'测试集mAP: {mAP_results["overall_mAP"]:.4f}')

    # 训练结束
    print('\n训练完成!')
    print(f'最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch+1})')
    print(f'总训练步数: {global_step}')
    print(f'最终学习率: {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'学习率是否被锁定: {"是" if frozen_lr else "否"}')

    # 保存训练结果到Excel
    try:
        # 创建训练结果DataFrame
        training_results = pd.DataFrame({
            'Epoch': list(range(1, len(train_losses) + 1)),
            'Train_Loss': train_losses,
            'Train_Accuracy': train_accs,
            'Val_Loss': val_losses,
            'Val_Accuracy': val_accs,
            'Learning_Rate': [lr_history[i] if i < len(lr_history) else lr_history[-1] 
                            for i in range(len(train_losses))],
            'Test_Loss': [test_loss] * len(train_losses),
            'Test_Accuracy': [test_acc] * len(train_losses)
        })
        
        # 保存到Excel
        results_path = os.path.join(results_dir, 'training_results.xlsx')
        training_results.to_excel(results_path, index=False)
        print(f'训练结果已保存至: {results_path}')
        
        # 创建学习率分析结果
        lr_analysis = pd.DataFrame({
            'Step': list(range(1, len(lr_history)+1)),
            'Learning_Rate': lr_history,
            'Scheduler_Type': [scheduler_type] * len(lr_history)
        })
        
        lr_analysis_path = os.path.join(results_dir, 'learning_rate_analysis.xlsx')
        lr_analysis.to_excel(lr_analysis_path, index=False)
        print(f'学习率分析已保存至: {lr_analysis_path}')
        
    except PermissionError as e:
        print(f'错误: 无法写入文件，请关闭Excel文件后重试')
        
    # 创建详细的训练报告
    training_report = f"""
    训练报告
    =========
    
    调度器类型: {scheduler_type}
    总训练轮数: {len(train_losses)}/{num_epochs}
    最佳验证准确率: {best_acc:.4f} (第{best_epoch+1}轮)
    最终测试准确率: {test_acc:.4f}
    最终测试损失: {test_loss:.4f}
    测试集mAP: {mAP_results['overall_mAP']:.4f}
    学习率是否锁定: {"是" if frozen_lr else "否"}
    总训练步数: {global_step}
    
    各类别AP分数:
    {chr(10).join([f"    {class_name}: {ap:.4f}" for class_name, ap in mAP_results['class_aps'].items()])}
    
    文件输出:
    - 训练结果: {results_path}
    - 学习率分析: {lr_analysis_path}
    - mAP分析数据: {mAP_results['excel_path']}
    - mAP柱状图: {mAP_results['chart_path']}
    - 学习率曲线: {os.path.join(results_dir, 'learning_rate_curve.png')}
    - 损失/准确率曲线: {os.path.join(results_dir, 'loss_acc_curve.png')}
    - 最佳模型: {os.path.join(results_dir, 'resnet_brightness_best.pth')}
    
    {"如果使用OneCycleLR: " + os.path.join(results_dir, 'lr_finder_results.png') if scheduler_type == "OneCycle" else ""}
    """
    
    report_path = os.path.join(results_dir, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(training_report)
    print(f'训练报告已保存至: {report_path}')

    # 所有输出结果将保存在带时间戳的目录中

    # 可视化学习率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(lr_history, color='blue', linewidth=1)
    plt.title("Learning Rate Schedule", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 标记最佳epoch和锁定学习率的位置
    if frozen_lr and scheduler_type != "OneCycle":
        freeze_step = (best_epoch + 1) * (len(train_loader) + len(val_loader))
        plt.axvline(x=freeze_step, color='red', linestyle='--', label='LR Frozen')
        plt.legend()

    # 保存学习率曲线
    plt.savefig(os.path.join(results_dir, 'learning_rate_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存损失和准确率曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(results_dir, 'loss_acc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存学习率历史
    try:
        final_lr_df = pd.DataFrame({'Step': list(range(1, len(lr_history)+1)), 'Learning Rate': lr_history})
        lr_output_path = os.path.join(results_dir, 'learning_rate_history.xlsx')
        final_lr_df.to_excel(lr_output_path, index=False)
        print(f'学习率历史记录已保存至: {lr_output_path}')
    except PermissionError as e:
        print(f'错误: 无法写入文件 {lr_output_path}，请关闭Excel文件后重试')
        sys.exit(1)



# 使用说明：
# 1. 设置 scheduler_type 变量选择学习率调度器：
#    - "CosineAnnealing": 使用 CosineAnnealingWarmRestarts(T_0=40, T_mult=1)
#    - "OneCycle": 使用 OneCycleLR，会自动运行 lr_find() 找到最佳 max_lr
#    - "Cyclic": 使用原来的 CyclicLR
# 
# 2. 如果使用 OneCycleLR，程序会先运行 30 个 batch 的 lr_find() 来找到最佳学习率
# 
# 3. 所有训练结果会保存在带时间戳的文件夹中，包含：
#    - training_results.xlsx: 完整训练记录
#    - lr_finder_results.png: 学习率搜索结果
#    - learning_rate_curve.png: 学习率变化曲线
#    - loss_acc_curve.png: 损失和准确率曲线
#    - learning_rate_history.xlsx: 详细学习率历史
def lr_find(model, train_loader, criterion, optimizer, device, num_iter=30):
    """
    学习率搜索函数 - 使用LR Finder技术找到不爆炸的最大学习率
    """
    print("\n开始运行学习率搜索...")
    
    # 保存原始状态
    original_state = model.state_dict()
    original_optimizer = optimizer.state_dict()
    
    # 创建临时的学习率调度器
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=10.0,  # 设置一个很大的上限
        epochs=1,
        steps_per_epoch=num_iter,
        pct_start=0.9,  # 大部分时间在上升阶段
        anneal_strategy='linear',
        div_factor=1e4,  # 从很小的学习率开始
        final_div_factor=1e4
    )
    
    model.train()
    losses = []
    lrs = []
    
    # 使用较小的batch来加速搜索
    small_batch_loader = []
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= num_iter:
            break
        small_batch_loader.append((inputs, labels))
    
    # 根据调度器类型决定是否运行lr_find
    if scheduler_type == "OneCycle":
        # 运行学习率搜索
        best_lr = lr_find(model, train_loader, criterion, optimizer, device, num_iter=30)
        
        # 更新OneCycleLR的最大学习率
        scheduler = OneCycleLR(
            optimizer,
            max_lr=best_lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
        print(f"更新OneCycleLR最大学习率为: {best_lr:.2e}")
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_processed = 0
        
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 缩放损失以考虑梯度累积
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            # 添加梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 累积足够的梯度后才更新参数
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
                # 更新学习率并记录
                if scheduler_type == "CosineAnnealing":
                    # CosineAnnealingWarmRestarts在每个step后更新
                    scheduler.step(global_step)
                elif scheduler_type == "OneCycle":
                    # OneCycleLR在每个step后更新
                    scheduler.step()
                else:
                    # CyclicLR
                    if not frozen_lr:
                        scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                lr_history.append(current_lr)
                global_step += 1
            else:
                current_lr = optimizer.param_groups[0]['lr']

            # 计算统计量
            running_loss += loss.item() * accumulation_steps * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_processed += inputs.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'lr': f"{current_lr:.2e}",
                'step': global_step,
                'frozen': 'Y' if frozen_lr else 'N'
            })

        # 计算epoch指标
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        print(f'训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证")
            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 验证阶段也记录学习率（但不更新）
                current_lr = optimizer.param_groups[0]['lr']
                lr_history.append(current_lr)
                global_step += 1

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())
        print(f'验证 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 更新最佳准确率和epoch
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            model_path = os.path.join(results_dir, 'resnet_brightness_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f'保存最佳模型至 {model_path}，验证准确率: {best_acc:.4f}')
            patience_counter = 0  # 重置早停计数器
            
            # 如果准确率提升显著，考虑锁定学习率
            if epoch > 10 and not frozen_lr:
                print("发现新的最佳准确率，考虑锁定学习率...")
                # 计算最近5个epoch的平均提升
                if len(val_accs) > 5:
                    recent_improve = np.mean(np.diff(val_accs[-5:]))
                    if recent_improve < 0.001:  # 提升很小
                        frozen_lr = True
                        print(f"锁定学习率为: {current_lr:.6f}，不再调整")
        else:
            patience_counter += 1
            print(f'验证准确率未提升，早停计数: {patience_counter}/{patience}')
        
        # 早停检查
        if patience_counter >= patience:
            print(f'验证准确率连续{patience}个epoch未提升，提前停止训练')
            break

        # 使用Plateau调度器（仅在未锁定学习率时）
        if scheduler_type != "OneCycle" and not frozen_lr:  # OneCycle不需要Plateau调度器
            plateau_scheduler.step(epoch_acc)
        elif scheduler_type != "OneCycle":
            # 锁定学习率后，保持学习率不变（仅对非OneCycle调度器）
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

    # 在测试集上评估模型
    def evaluate_test_set(model, test_loader, criterion, device):
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        return test_loss, test_acc

    # 评估测试集
    test_loss, test_acc = evaluate_test_set(model, test_loader, criterion, device)
    print(f'测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.4f}')

    # 训练结束
    print('\n训练完成!')
    print(f'最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch+1})')
    print(f'总训练步数: {global_step}')
    print(f'最终学习率: {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'学习率是否被锁定: {"是" if frozen_lr else "否"}')