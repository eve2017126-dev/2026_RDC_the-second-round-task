import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=10000):
        """初始化模型参数"""
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.cost_history = []
    
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        
        for i in range(self.n_iterations):
            # 计算预测值
            y_pred = np.dot(X, self.weights)
            
            # 计算损失（均方误差）
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
            
            # 计算梯度
            gradient = (1/n_samples) * np.dot(X.T, (y_pred - y))
            
            # 更新参数
            self.weights -= self.learning_rate * gradient
        
        return self
    
    def predict(self, X):
        """预测新数据"""
        return np.dot(X, self.weights)