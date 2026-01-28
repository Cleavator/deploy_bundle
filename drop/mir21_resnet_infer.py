
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# 设置随机种子以确保结果可重现
def set_random_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # 关闭cudnn的自动优化以确保结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # 使用预训练的ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # 替换最后的全连接层
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def load_model(model_path, num_classes=8):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=num_classes)
    
    if os.path.exists(model_path):
        try:
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=device)
            
            # 处理可能保存为整个模型或仅state_dict的情况
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
                
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file not found: {model_path}")
        return None

def predict_mir21_concentration_from_array(img: np.ndarray) -> float:
    """
    加载训练好的 ResNet 权重，对单张图片做推理，返回 miR-21 浓度预测值。
    """
    # 预处理 transform
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 转换图片格式
    if not isinstance(img, Image.Image):
        img_pil = Image.fromarray(img)
    else:
        img_pil = img
        
    # 预处理
    img_tensor = val_test_transform(img_pil).unsqueeze(0) # add batch dim
    
    # 加载模型
    # 注意：这里需要指定模型路径。由于没有明确的 miR-21 模型路径，我们假设它在 drop 目录下或者通过某种规则找到
    # 这里假设有一个 best_model.pth
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 尝试寻找 best_model.pth 或 resnet_brightness_best.pth
    model_path = None
    target_files = ["best_model.pth", "resnet_brightness_best.pth"]
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in target_files:
                model_path = os.path.join(root, file)
                break
        if model_path:
            break
            
    if not model_path:
        print("Warning: No best_model.pth found for miR-21 prediction.")
        return 0.0 # 默认值
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path)
    
    if model is None:
        return 0.0
        
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        class_idx = preds.item()
        
    # 映射 class_idx 到浓度
    # 假设 classes = ['1fM', '10fM', '100fM', '1pM', '10pM', '100pM', '1nM', 'blank']
    # 注意：需要确保这个顺序与训练时一致
    classes = ['1fM', '10fM', '100fM', '1pM', '10pM', '100pM', '1nM', 'blank']
    # 按字母顺序排序，因为 ImageFolder 通常是按字母顺序
    classes.sort() 
    
    pred_class = classes[class_idx]
    
    # 解析浓度值
    import re
    if 'blank' in pred_class.lower():
        return 0.0
    
    digit_match = re.search(r'(\d+\.?\d*)', pred_class)
    unit_match = re.search(r'(fM|pM|nM)', pred_class, re.IGNORECASE)
    
    if digit_match and unit_match:
        number = float(digit_match.group(1))
        unit = unit_match.group(1).lower()
        if unit == 'fm':
            return number * 1e-15
        elif unit == 'pm':
            return number * 1e-12
        elif unit == 'nm':
            return number * 1e-9
            
    return 0.0
