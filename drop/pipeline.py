import os
import pandas as pd
import pickle  # 使用与保存相同的库
import argparse
from droplet_analyzer_cellpose_v3 import batch_process_droplets  # 引入液滴分析函数

def load_knn_model(model_path):
    """加载训练好的KNN模型"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model']  # 返回模型对象

def process_image(image_path, output_dir, model_type='21N'):
    """
    处理图片并进行预测
    image_path: 输入图片路径
    output_dir: 输出目录
    model_type: 选择模型 '21N' 或 '92a'
    """
    # 步骤 0: 提前加载两个模型（性能优化）
    print("加载模型...")
    # 使用正确的模型路径和名称
    model_21n_path = './KNN-21N_model.pkl'  # 当前目录下的模型文件
    model_92a_path = './KNN-92a_model.pkl'  # 当前目录下的模型文件
    
    with open(model_21n_path, 'rb') as f:
        model_21n_data = pickle.load(f)
        model_21n = model_21n_data['model']
        scaler_21n = model_21n_data['scaler']
        feature_cols_21n = model_21n_data['feature_cols']
        
    with open(model_92a_path, 'rb') as f:
        model_92a_data = pickle.load(f)
        model_92a = model_92a_data['model']
        scaler_92a = model_92a_data['scaler']
        feature_cols_92a = model_92a_data['feature_cols']
    
    print("模型加载完成")

    # 步骤 1: 液滴分割与特征提取
    print(f"处理图片: {image_path}")
    feature_data = batch_process_droplets(image_path, output_dir)  # 获取液滴特征

    # 保存提取的特征数据到 CSV
    feature_csv = os.path.join(output_dir, f'{os.path.basename(image_path)}_features.csv')
    feature_data.to_csv(feature_csv, index=False)
    print(f"液滴特征已保存至: {feature_csv}")

    # 步骤 2: 根据颜色选择分类器并预测
    predictions = []
    for index, row in feature_data.iterrows():
        # 准备特征数据
        features = row[['mean_Gray', 'mean_Red', 'mean_Green', 'mean_Blue']].values.reshape(1, -1)
        
        # 判断液滴颜色并选择分类器
        if row['mean_Red'] > row['mean_Green']:  # 红色液滴
            # 使用预加载的 KNN-21N 模型
            scaled_features = scaler_21n.transform(features)  # 标准化特征
            prediction = model_21n.predict(scaled_features)  # 输入特征进行预测
        else:  # 绿色液滴
            # 使用预加载的 KNN-92a 模型
            scaled_features = scaler_92a.transform(features)  # 标准化特征
            prediction = model_92a.predict(scaled_features)  # 输入特征进行预测
        
        predictions.append(prediction[0])

    # 步骤 3: 输出预测结果
    result_df = pd.DataFrame({
        'Image': [os.path.basename(image_path)] * len(predictions),
        'Prediction': predictions,
        'Mean_Red': feature_data['mean_Red'],
        'Mean_Green': feature_data['mean_Green'],
        'Mean_Blue': feature_data['mean_Blue'],
    })

    result_csv = os.path.join(output_dir, f'{os.path.basename(image_path)}_predictions.csv')
    result_df.to_csv(result_csv, index=False)
    print(f"预测结果已保存至: {result_csv}")

    return result_df

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="图片处理和分类预测")
    parser.add_argument('image', type=str, help="输入图片路径")
    parser.add_argument('--output_dir', type=str, default='./output', help="输出结果的目录")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理图像并进行分类
    process_image(args.image, args.output_dir)

if __name__ == "__main__":
    main()
