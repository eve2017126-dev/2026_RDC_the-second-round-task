import numpy as np

def calculate_mse(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred)**2)

def calculate_rmse(y_true, y_pred):
    """计算均方根误差"""
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    """计算R²值"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    mse = calculate_mse(y_test, y_pred)
    rmse = calculate_rmse(y_test, y_pred)
    r2 = calculate_r2(y_test, y_pred)
    return mse, rmse, r2