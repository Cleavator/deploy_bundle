import joblib  # 或者 pickle

# 假设已经训练好的模型是 best_knn
joblib.dump(best_knn, 'knn_model_21n.pkl')  # 保存为 knn_model_21n.pkl 文件
