import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import json
import math
import subprocess
import sys

class BrightnessExtractor:
    def __init__(self):
        self.min_radius = 5
        self.max_radius = 30
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
    
    def detect_balls(self, image_np):
        """检测图像中的圆形物体"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20, 
            param1=self.canny_threshold1, param2=self.canny_threshold2, 
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        return circles
    
    def extract_brightness(self, image_path):
        """从图像中提取亮度特征"""
        try:
            # 读取图像
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            
            # 转换为BGR格式用于OpenCV处理
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
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
            
            # 如果检测到球体，使用球体的平均亮度；否则使用整个图像的平均亮度
            if ball_brightness:
                avg_brightness = np.mean(ball_brightness)
            else:
                avg_brightness = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
            
            return avg_brightness
        except Exception as e:
            print(f"提取图像亮度时出错 {image_path}: {str(e)}")
            return None


def predict_mir92a_concentration_from_array(img: np.ndarray) -> float:
    """
    输入原始图像数组，返回预测的 miR-92a 浓度（单位按你训练时的单位）。
    """
    # 复用 BrightnessExtractor
    extractor = BrightnessExtractor()
    
    # 转换为BGR (BrightnessExtractor 内部期望 RGB 转 BGR)
    # img 是 RGB (from Gradio/PIL)
    if img.ndim == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img # 假设已经是 BGR 或灰度
        
    # 提取亮度
    circles = extractor.detect_balls(img_bgr)
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
        
    # 使用线性回归模型预测
    # 注意：这里需要加载训练好的 lr_model
    # 由于 lr_model 没有被持久化保存为独立文件（脚本里是每次重新训练），
    # 我们这里使用一个硬编码的简单公式或者占位符，或者尝试加载之前的 fit_results.json
    # 假设我们有一个简单的线性关系：Concentration = a * Brightness + b
    # 或者我们暂时返回亮度值作为代理指标
    
    # 尝试加载 fit_results.json
    try:
        results_path = os.path.join(os.path.dirname(__file__), 'brightness_analysis_results', 'fit_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                fit_results = json.load(f)
                slope = fit_results.get('slope', 0)
                intercept = fit_results.get('intercept', 0)
                # 浓度是对数关系：Brightness = slope * log(Conc) + intercept
                # => log(Conc) = (Brightness - intercept) / slope
                # => Conc = 10^((Brightness - intercept) / slope)
                if slope != 0:
                    log_conc = (avg_brightness - intercept) / slope
                    conc = 10 ** log_conc
                    return conc
    except Exception as e:
        print(f"Error loading fit results: {e}")
        
    # 如果无法加载模型，返回亮度值（负数表示这是原始亮度而非浓度）
    return -avg_brightness 


class BrightnessConcentrationAnalyzer:
    def __init__(self, data_dir, results_dir=None):
        self.data_dir = data_dir
        self.brightness_extractor = BrightnessExtractor()
        self.classes = ['blank', '1fM', '10fM', '100fM', '1pM', '10pM', '100pM', '1nM']
        self.concentration_mapping = self._create_concentration_mapping()
        self.results_dir = results_dir or os.path.join(data_dir, 'brightness_analysis_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 模型和结果存储
        self.lr_model = None
        self.fit_results = None
        self.avg_brightness_by_class = None
        self.all_brightness_by_class = None  # 存储每个类别的所有亮度值，用于计算标准偏差
        self.detection_limit = None  # 检出限
        
        # 训练脚本路径
        self.train_script_path = os.path.join(self.data_dir, 'train_resnet 1.2.7.py')
        if not os.path.exists(self.train_script_path):
            print(f"警告: 未找到训练脚本 '{self.train_script_path}'")
        
    def call_train_script(self, image_path, output_dir=None):
        """调用训练脚本进行图像分类
        
        参数:
            image_path: 输入图像的路径
            output_dir: 输出结果保存目录（可选）
            
        返回:
            str: 分类结果类别名称，或None（如果分类失败）
        """
        if not os.path.exists(self.train_script_path):
            print(f"错误: 未找到训练脚本 '{self.train_script_path}'")
            return None
            
        if not os.path.exists(image_path):
            print(f"错误: 未找到图像文件 '{image_path}'")
            return None
            
        # 设置输出目录
        if output_dir is None:
            output_dir = self.results_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            # 调用训练脚本进行分类
            # 假设训练脚本接受--image和--output参数
            cmd = [
                sys.executable, self.train_script_path,
                '--image', image_path,
                '--output', output_dir,
                '--mode', 'predict'
            ]
            
            # 执行命令并捕获输出
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(self.train_script_path)
            )
            
            # 检查命令执行是否成功
            if result.returncode != 0:
                print(f"调用训练脚本时出错: {result.stderr}")
                return None
            
            # 解析输出结果
            # 假设训练脚本输出中包含"分类结果: X"这样的格式
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if '分类结果:' in line or 'classification result:' in line.lower():
                    # 提取分类结果
                    class_name = line.split(':')[1].strip()
                    print(f"训练脚本分类结果: {class_name}")
                    return class_name
                elif line.strip() in self.classes:
                    # 如果输出直接包含类别名称
                    print(f"训练脚本分类结果: {line.strip()}")
                    return line.strip()
            
            # 如果没有找到预期的输出格式，返回最后一行
            if output_lines:
                last_line = output_lines[-1].strip()
                if last_line in self.classes:
                    print(f"训练脚本分类结果: {last_line}")
                    return last_line
            
            print(f"无法从训练脚本输出中解析分类结果: {result.stdout}")
            return None
            
        except Exception as e:
            print(f"调用训练脚本时发生异常: {str(e)}")
            return None
        
    def _create_concentration_mapping(self):
        """创建类别名称到实际浓度值的映射"""
        mapping = {}
        print("浓度的负对数值计算结果：")
        
        # 测试用例 - 直接设置浓度值以确保计算正确
        test_concentrations = {
            'blank': 0.0,
            '1fM': 1e-15,
            '10fM': 10e-15,
            '100fM': 100e-15,
            '1pM': 1e-12,
            '10pM': 10e-12,
            '100pM': 100e-12,
            '1nM': 1e-9
        }
        
        for class_name in self.classes:
            # 首先尝试使用测试用例中的浓度值
            if class_name in test_concentrations:
                concentration = test_concentrations[class_name]
            # 然后尝试通过文件夹名解析
            elif 'blank' in class_name.lower():
                concentration = 0.0
            else:
                # 使用正则表达式更可靠地提取数字和单位
                import re
                # 匹配数字部分（可能包含小数点）
                digit_match = re.search(r'(\d+\.?\d*)', class_name)
                # 匹配单位部分
                unit_match = re.search(r'(fM|pM|nM)', class_name, re.IGNORECASE)
                
                if digit_match and unit_match:
                    try:
                        number = float(digit_match.group(1))
                        unit = unit_match.group(1).lower()
                        
                        if unit == 'fm':
                            concentration = number * 1e-15
                        elif unit == 'pm':
                            concentration = number * 1e-12
                        elif unit == 'nm':
                            concentration = number * 1e-9
                        else:
                            concentration = 0.0
                    except:
                        concentration = 0.0
                else:
                    concentration = 0.0
            
            mapping[class_name] = concentration
            
            # 计算并输出负对数值（跳过blank样本）
            if concentration > 0:
                try:
                    negative_log = -math.log10(concentration)
                    print(f"浓度 {class_name}: 浓度值 = {concentration:.2e} M, 负对数值 = {negative_log:.4f}")
                except:
                    print(f"浓度 {class_name}: 无法计算负对数值")
        return mapping
    
    def analyze_dataset_brightness(self, max_samples_per_class=100):
        """分析数据集中每个类别的平均亮度"""
        avg_brightness_by_class = {}
        all_brightness_by_class = {}
        
        supported_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        
        print("正在分析每个类别的平均亮度...")
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 类别文件夹 {class_name} 不存在")
                continue
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(supported_extensions)]
            if not image_files:
                print(f"警告: 类别 {class_name} 中没有找到有效的图像文件")
                continue
            
            # 限制样本数量以提高效率
            if len(image_files) > max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            brightness_values = []
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                brightness = self.brightness_extractor.extract_brightness(image_path)
                if brightness is not None:
                    brightness_values.append(brightness)
            
            if brightness_values:
                avg_brightness = np.mean(brightness_values)
                avg_brightness_by_class[class_name] = avg_brightness
                all_brightness_by_class[class_name] = brightness_values
                print(f"类别 {class_name} 的平均亮度: {avg_brightness:.2f} (n={len(brightness_values)})")
                print(f"类别 {class_name} 的亮度标准偏差: {np.std(brightness_values):.2f}")
            else:
                print(f"警告: 无法从类别 {class_name} 中提取有效的亮度值")
        
        self.avg_brightness_by_class = avg_brightness_by_class
        self.all_brightness_by_class = all_brightness_by_class  # 保存所有亮度值
        
        # 保存亮度数据到Excel
        if avg_brightness_by_class:
            brightness_data = []
            for class_name, brightness in avg_brightness_by_class.items():
                brightness_data.append({
                    'Class': class_name,
                    'Concentration': self.concentration_mapping[class_name],
                    'Avg_Brightness': brightness,
                    'Std_Brightness': np.std(all_brightness_by_class[class_name]) if class_name in all_brightness_by_class else None,
                    'Sample_Count': len(all_brightness_by_class[class_name]) if class_name in all_brightness_by_class else 0
                })
            
            df = pd.DataFrame(brightness_data)
            excel_path = os.path.join(self.results_dir, 'brightness_by_class.xlsx')
            df.to_excel(excel_path, index=False)
            print(f"亮度数据已保存至: {excel_path}")
            
            # 保存原始亮度数据到单独的Excel文件
            self._save_raw_brightness_data(all_brightness_by_class)
        
        return avg_brightness_by_class
    
    def fit_brightness_concentration_model(self):
        """拟合浓度负对数和亮度之间的线性模型"""
        if not self.avg_brightness_by_class:
            self.analyze_dataset_brightness()
            if not self.avg_brightness_by_class:
                raise ValueError("无法提取足够的亮度数据来拟合模型")
        
        # 创建浓度映射
        if not hasattr(self, 'concentration_mapping') or not self.concentration_mapping:
            self.concentration_mapping = self._create_concentration_mapping()
        
        # 准备训练数据，区分包含和不包含blank的数据集
        X_all = []  # 包含所有类别的浓度负对数值特征
        y_all = []  # 包含所有类别的亮度目标
        class_names_all = []
        
        # 准备只包含7个浓度类别的数据用于绘图和分析
        X_7conc = []  # 只包含7个浓度类别的浓度负对数值特征
        y_7conc = []  # 只包含7个浓度类别的亮度目标
        class_names_7conc = []
        
        for class_name, brightness in self.avg_brightness_by_class.items():
            concentration = self.concentration_mapping[class_name]
            # 计算浓度的负对数，注意要避免log(0)
            neg_log_concentration = -np.log10(concentration) if concentration > 0 else 0
            
            # 添加到包含所有类别的数据集
            X_all.append([neg_log_concentration])
            y_all.append(brightness)
            class_names_all.append(class_name)
            
            # 添加到只包含7个浓度类别的数据集（排除blank）
            if class_name != 'blank':
                X_7conc.append([neg_log_concentration])
                y_7conc.append(brightness)
                class_names_7conc.append(class_name)
        
        # 转换为numpy数组
        X_7conc = np.array(X_7conc)
        y_7conc = np.array(y_7conc)
        
        # 创建并训练线性回归模型，使用只包含7个浓度类别的数据集
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_7conc, y_7conc)
        
        # 计算预测值和评估指标
        y_pred_7conc = self.lr_model.predict(X_7conc)
        r_squared = r2_score(y_7conc, y_pred_7conc)
        
        # 保存拟合结果
        self.fit_results = {
            'slope': float(self.lr_model.coef_[0]),
            'intercept': float(self.lr_model.intercept_),
            'r_squared': float(r_squared),
            'equation': f"y = {self.lr_model.coef_[0]:.4f}x + {self.lr_model.intercept_:.4f}"
        }
        
        # 可视化拟合结果，使用只包含7个浓度类别的数据集
        self._visualize_fit(X_7conc, y_7conc, y_pred_7conc, class_names_7conc)
        
        # 保存模型参数和结果，使用只包含7个浓度类别的数据集
        self._save_model_results(X_7conc, y_7conc, y_pred_7conc, class_names_7conc)
        
        print(f"拟合方程: {self.fit_results['equation']}")
        print(f"决定系数 (R²): {r_squared:.4f}")
        
        return self.fit_results
    
    def _visualize_fit(self, X, y, y_pred, class_names):
        """可视化浓度负对数和亮度之间的拟合关系"""
        plt.figure(figsize=(12, 8))
        
        # 绘制散点图
        plt.scatter(X, y, color='blue', alpha=0.8, label='实际数据点')
        
        # 添加类别标签
        for i, class_name in enumerate(class_names):
            plt.annotate(class_name, (X[i], y[i]), fontsize=9, 
                         xytext=(5, 5), textcoords='offset points')
        
        # 绘制拟合直线
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = self.lr_model.predict(X_line)
        plt.plot(X_line, y_line, 'r-', linewidth=2, 
                 label=f'线性拟合 (R²={self.fit_results["r_squared"]:.4f})')
        
        # 添加标签和标题
        plt.xlabel('浓度的负对数 (-log₁₀[浓度])', fontsize=12)
        plt.ylabel('平均亮度', fontsize=12)
        plt.title('浓度负对数与图像亮度的线性关系拟合', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        plot_path = os.path.join(self.results_dir, 'brightness_neg_log_concentration_fit.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"拟合关系图已保存至: {plot_path}")
    
    def _save_raw_brightness_data(self, all_brightness_by_class):
        """保存每个类别的原始亮度数据"""
        try:
            # 创建一个大的数据框，包含所有类别的原始亮度数据
            all_data = []
            for class_name, brightness_values in all_brightness_by_class.items():
                concentration = self.concentration_mapping[class_name]
                for brightness in brightness_values:
                    all_data.append({
                        'Class': class_name,
                        'Concentration': concentration,
                        'Brightness': brightness
                    })
            
            df = pd.DataFrame(all_data)
            raw_data_path = os.path.join(self.results_dir, 'raw_brightness_data.xlsx')
            df.to_excel(raw_data_path, index=False)
            print(f"原始亮度数据已保存至: {raw_data_path}")
        except Exception as e:
            print(f"保存原始亮度数据时出错: {str(e)}")
    
    def calculate_detection_limit(self):
        """计算检出限 (LOD)
        
        使用公式: LOD = 3σ/S
        其中: σ 是空白样本的标准偏差，S 是校准曲线的斜率
        """
        if self.lr_model is None:
            raise ValueError("请先拟合模型")
        
        if self.all_brightness_by_class is None:
            raise ValueError("请先分析数据集亮度")
        
        # 检查是否有空白样本数据
        if 'blank' in self.all_brightness_by_class and len(self.all_brightness_by_class['blank']) > 0:
            # 使用空白样本计算标准偏差
            blank_brightness_values = self.all_brightness_by_class['blank']
            blank_std = np.std(blank_brightness_values)
            n_blank = len(blank_brightness_values)
            print(f"空白样本统计: 样本数={n_blank}, 标准偏差σ={blank_std:.4f}")
        else:
            # 如果没有空白样本，尝试使用最低浓度的样本作为替代
            print("警告: 未找到足够的空白样本数据，尝试使用最低浓度样本作为替代")
            # 找到最低浓度的类别
            sorted_classes = sorted(self.concentration_mapping.items(), key=lambda x: x[1])
            for class_name, _ in sorted_classes:
                if class_name != 'blank' and class_name in self.all_brightness_by_class and len(self.all_brightness_by_class[class_name]) > 0:
                    low_conc_values = self.all_brightness_by_class[class_name]
                    blank_std = np.std(low_conc_values)
                    n_blank = len(low_conc_values)
                    print(f"使用 {class_name} 作为近空白样本: 样本数={n_blank}, 标准偏差σ={blank_std:.4f}")
                    break
            else:
                raise ValueError("无法找到足够的样本数据来计算标准偏差")
        
        # 获取校准曲线的斜率
        slope = abs(self.lr_model.coef_[0])  # 使用绝对值
        
        # 计算检出限 (LOD = 3σ/S)
        lod = (3 * blank_std) / slope
        
        # 确保检出限为正数
        lod = max(0, lod)
        
        self.detection_limit = lod
        
        print(f"检出限 (LOD): {lod:.2e} M")
        return lod
    
    def _save_model_results(self, X, y, y_pred, class_names):
        """保存模型参数和结果数据"""
        # 保存模型参数
        model_params = {
            'slope': float(self.lr_model.coef_[0]),
            'intercept': float(self.lr_model.intercept_),
            'r_squared': float(self.fit_results['r_squared']),
            'detection_limit': float(self.detection_limit) if self.detection_limit is not None else None
        }
        
        model_path = os.path.join(self.results_dir, 'brightness_neg_log_concentration_model.json')
        with open(model_path, 'w') as f:
            json.dump(model_params, f, indent=2)
        print(f"模型参数已保存至: {model_path}")
        
        # 保存详细结果到Excel
        results_data = []
        for i, class_name in enumerate(class_names):
            # 将预测的负对数值转换回浓度值
            concentration = self.concentration_mapping[class_name]
            results_data.append({
                'Class': class_name,
                'Avg_Brightness': float(X[i][0]),
                'Concentration': concentration,
                'True_Neg_Log_Concentration': float(y[i]),
                'Predicted_Neg_Log_Concentration': float(y_pred[i]),
                'Absolute_Error': float(abs(y[i] - y_pred[i])),
                'Relative_Error': float(abs(y[i] - y_pred[i]) / (y[i] + 1e-15)) if y[i] != 0 else 0.0
            })
            
        df = pd.DataFrame(results_data)
        excel_path = os.path.join(self.results_dir, 'brightness_neg_log_concentration_results.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"详细结果已保存至: {excel_path}")
        
        # 生成分析报告
        report_path = os.path.join(self.results_dir, 'brightness_neg_log_concentration_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("亮度-浓度负对数关系分析报告\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"拟合方程: y = {self.lr_model.coef_[0]:.4f}x + {self.lr_model.intercept_:.4f}\n")
            f.write(f"决定系数 (R²): {self.fit_results['r_squared']:.4f}\n")
            if self.detection_limit is not None:
                f.write(f"检出限 (LOD): {self.detection_limit:.2e} M\n\n")
            else:
                f.write("\n")
            f.write("数据点详情:\n")
            for i, class_name in enumerate(class_names):
                concentration = self.concentration_mapping[class_name]
                f.write(f"- {class_name}: 亮度={X[i][0]:.2f}, 浓度={concentration:.2e}M, 浓度负对数={y[i]:.2f}\n")
        print(f"分析报告已保存至: {report_path}")
    
    def load_model(self, model_path=None):
        """加载已保存的模型参数"""
        if model_path is None:
            model_path = os.path.join(self.results_dir, 'brightness_neg_log_concentration_model.json')
            
        if not os.path.exists(model_path):
            # 尝试加载旧的模型文件作为后备选项
            old_model_path = os.path.join(self.results_dir, 'brightness_concentration_model.json')
            if os.path.exists(old_model_path):
                model_path = old_model_path
                print("警告: 使用旧格式的模型文件")
            else:
                raise FileNotFoundError(f"未找到模型文件: {model_path}")
            
        with open(model_path, 'r') as f:
            model_params = json.load(f)
            
        self.lr_model = LinearRegression()
        self.lr_model.coef_ = np.array([model_params['slope']])
        self.lr_model.intercept_ = model_params['intercept']
        self.fit_results = model_params
        
        print(f"模型已从 {model_path} 加载")
        return self.lr_model
    
    def predict_concentration(self, image_path):
        """预测新图像的浓度"""
        if self.lr_model is None:
            try:
                self.load_model()
            except FileNotFoundError:
                print("未找到已保存的模型，正在重新拟合...")
                self.fit_brightness_concentration_model()
        
        # 提取图像亮度
        brightness = self.brightness_extractor.extract_brightness(image_path)
        if brightness is None:
            return None, None, "无法从图像中提取亮度特征"
        
        # 从亮度预测浓度的负对数值
        # 模型是 y = slope * x + intercept，其中 y 是亮度，x 是浓度负对数
        # 所以 x = (y - intercept) / slope
        if self.lr_model.coef_[0] == 0:
            return None, None, "模型斜率为0，无法预测浓度"
            
        neg_log_concentration = (brightness - self.lr_model.intercept_) / self.lr_model.coef_[0]
        
        # 将负对数值转换回浓度值
        concentration = 10 ** (-neg_log_concentration)
        
        # 计算浓度范围（基于拟合的不确定性）
        # 这里使用简化的方法计算不确定性，实际应用中可能需要更复杂的统计方法
        # 假设预测的相对误差为10%
        lower_neg_log = neg_log_concentration * 1.1  # 负对数值增加10%意味着浓度降低
        upper_neg_log = neg_log_concentration * 0.9  # 负对数值减少10%意味着浓度增加
        lower_bound = 10 ** (-lower_neg_log)
        upper_bound = 10 ** (-upper_neg_log)
        
        # 确保浓度非负
        concentration = max(0, concentration)
        lower_bound = max(0, lower_bound)
        upper_bound = max(0, upper_bound)
        
        # 检查是否低于检出限
        lod_info = ""
        if self.detection_limit is not None and concentration < self.detection_limit:
            lod_info = "(低于检出限)"
        
        return concentration, neg_log_concentration, lod_info
    
    def classify_image(self, image_path):
        """将输入的图像分类到八个文件夹中的一个，并预测其浓度
        
        参数:
            image_path: 输入图像的路径
        
        返回:
            tuple: (分类结果, 预测浓度, 预测浓度负对数值, 检出限信息)
        """
        # 提取图像亮度 - 用于浓度预测
        brightness = self.brightness_extractor.extract_brightness(image_path)
        if brightness is None:
            return None, None, None, "无法从图像中提取亮度特征"
        
        # 调用训练脚本进行分类
        best_class = self.call_train_script(image_path)
        
        # 如果训练脚本分类失败，不再回退到基于亮度的分类
        if best_class is None:
            print("训练脚本分类失败")
            return None, None, None, "训练脚本分类失败"
        
        # 使用亮度分析预测浓度
        concentration, neg_log_concentration, lod_info = self.predict_concentration(image_path)
        
        return best_class, concentration, neg_log_concentration, lod_info

import sys
import argparse

if __name__ == '__main__':
    # 设置数据目录
    data_dir = "d:/green-miR-92a"
    
    # 创建分析器实例
    analyzer = BrightnessConcentrationAnalyzer(data_dir)
    
    # 分类逻辑现在通过调用外部训练脚本实现
    
    # 分析数据集亮度
    analyzer.analyze_dataset_brightness(max_samples_per_class=50)  # 限制每个类别的样本数量以加速分析
    
    # 拟合亮度-浓度模型
    analyzer.fit_brightness_concentration_model()
    
    # 计算检出限
    try:
        analyzer.calculate_detection_limit()
    except Exception as e:
        print(f"计算检出限时出错: {str(e)}")
    
    print("\n分析完成！所有结果已保存到 'brightness_analysis_results' 文件夹中。")
    
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='图像分类和浓度预测工具')
    parser.add_argument('--image', type=str, help='输入图像的路径')
    parser.add_argument('--folder', type=str, help='输入文件夹的路径')
    parser.add_argument('--interactive', action='store_true', help='启动交互式模式')

    # 解析命令行参数
    args = parser.parse_args()

    # 处理命令行参数
    if args.image:
        # 处理单个图像文件
        if os.path.isfile(args.image):
            print("\n======= 图片分类与浓度预测功能 =======")
            print(f"处理文件: {args.image}")
            try:
                # 执行分类和浓度预测
                best_class, concentration, neg_log_concentration, lod_info = analyzer.classify_image(args.image)
                
                if best_class is None:
                    print(f"分类失败: {concentration}")
                else:
                    print(f"图片分类结果: {best_class}")
                    print(f"预测浓度: {concentration:.2e} M {lod_info}")
                    print(f"预测浓度负对数值: {neg_log_concentration:.2f}")
                    print(f"建议保存到文件夹: {best_class}")
            except Exception as e:
                print(f"处理图像时出错: {str(e)}")
        else:
            print(f"错误: 找不到文件 '{args.image}'")
    elif args.folder:
        # 处理文件夹
        if os.path.isdir(args.folder):
            print("\n======= 图片分类与浓度预测功能 =======")
            print(f"处理文件夹: {args.folder}")
            # 获取文件夹中的所有图像文件
            image_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for root, dirs, files in os.walk(args.folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                print(f"错误: 在文件夹 '{args.folder}' 中未找到图像文件")
            else:
                print(f"找到 {len(image_files)} 个图像文件，开始处理...")
                
                # 处理每个图像文件
                for image_file in image_files:
                    print(f"\n处理文件: {image_file}")
                    try:
                        # 执行分类和浓度预测
                        best_class, concentration, neg_log_concentration, lod_info = analyzer.classify_image(image_file)
                        
                        if best_class is None:
                            print(f"分类失败: {concentration}")
                        else:
                            print(f"图片分类结果: {best_class}")
                            print(f"预测浓度: {concentration:.2e} M {lod_info}")
                            print(f"预测浓度负对数值: {neg_log_concentration:.2f}")
                            print(f"建议保存到文件夹: {best_class}")
                    except Exception as e:
                        print(f"处理图像时出错: {str(e)}")
                        
                print(f"\n文件夹 '{args.folder}' 处理完成。")
        else:
            print(f"错误: 找不到文件夹 '{args.folder}'")
    else:
        # 默认启动交互式模式
        print("\n======= 图片分类与浓度预测功能 =======")
        print("输入图像路径或文件夹路径进行分类和浓度预测，或输入 'exit' 退出")
        print("提示: 您也可以使用命令行参数如 --image 或 --folder 直接指定路径")
        
        while True:
            user_input = input("请输入图像路径或文件夹路径: ")
            if user_input.lower() == 'exit':
                break
            
            # 检查是否为文件夹路径
            if os.path.isdir(user_input):
                print(f"\n处理文件夹: {user_input}")
                # 获取文件夹中的所有图像文件
                image_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']
                image_files = []
                
                for root, dirs, files in os.walk(user_input):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(root, file))
                
                if not image_files:
                    print(f"错误: 在文件夹 '{user_input}' 中未找到图像文件")
                    continue
                
                print(f"找到 {len(image_files)} 个图像文件，开始处理...")
                
                # 处理每个图像文件
                for image_file in image_files:
                    print(f"\n处理文件: {image_file}")
                    try:
                        # 执行分类和浓度预测
                        best_class, concentration, neg_log_concentration, lod_info = analyzer.classify_image(image_file)
                        
                        if best_class is None:
                            print(f"分类失败: {concentration}")
                        else:
                            print(f"图片分类结果: {best_class}")
                            print(f"预测浓度: {concentration:.2e} M {lod_info}")
                            print(f"预测浓度负对数值: {neg_log_concentration:.2f}")
                            print(f"建议保存到文件夹: {best_class}")
                    except Exception as e:
                        print(f"处理图像时出错: {str(e)}")
                        
                print(f"\n文件夹 '{user_input}' 处理完成。")
            
            # 检查是否为文件路径
            elif os.path.isfile(user_input):
                try:
                    # 执行分类和浓度预测
                    best_class, concentration, neg_log_concentration, lod_info = analyzer.classify_image(user_input)
                    
                    if best_class is None:
                        print(f"分类失败: {concentration}")
                    else:
                        print(f"\n图片分类结果: {best_class}")
                        print(f"预测浓度: {concentration:.2e} M {lod_info}")
                        print(f"预测浓度负对数值: {neg_log_concentration:.2f}")
                        print(f"建议保存到文件夹: {best_class}")
                except Exception as e:
                    print(f"处理图像时出错: {str(e)}")
            
            else:
                print(f"错误: 找不到文件或文件夹 '{user_input}'")

    print("\n程序已退出。")