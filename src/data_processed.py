import numpy as np
import pandas as pd

def load_data(file_path):
    """加载原始数据"""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """预处理数据：处理分类特征，添加偏置项，标准化"""
    # 独热编码分类特征
    data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    # 分离特征和目标变量
    X = data_encoded.drop('charges', axis=1).values
    y = data_encoded['charges'].values.reshape(-1, 1)
    
    # 特征标准化
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    
    # 添加偏置项
    X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))
    
    return X_normalized, y, mean, std

def train_test_split(X, y, test_size=0.2, random_state=42):
    """分割训练集和测试集"""
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(test_size * X.shape[0])
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]