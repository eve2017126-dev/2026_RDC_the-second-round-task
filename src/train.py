import numpy as np
from .data_processed import load_data, preprocess_data, train_test_split
from .model import LinearRegression

def train_model(data_path):
    """训练模型的完整流程"""
    # 加载数据
    data = load_data(data_path)
    
    # 预处理数据
    X, y, mean, std = preprocess_data(data)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # 初始化模型
    model = LinearRegression(learning_rate=0.01, n_iterations=10000)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, mean, std