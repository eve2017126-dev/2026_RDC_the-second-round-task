import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False
from src.train import train_model

def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    weights = model.weights.flatten()
    feature_importance = abs(weights[1:])  # 排除偏置项
    
    print("\n" + "="*50)
    print("特征重要性分析")
    print("="*50)
    
    # 排序并显示特征重要性
    sorted_indices = np.argsort(feature_importance)[::-1]
    for i, idx in enumerate(sorted_indices):
        if idx < len(feature_names):
            importance_score = feature_importance[idx] / np.sum(feature_importance) * 100
            print(f"{i+1:2d}. {feature_names[idx]:15s} | 重要性: {importance_score:6.2f}% | 权重: {weights[idx+1]:8.4f}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance[sorted_indices])
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in sorted_indices], rotation=45)
    plt.xlabel('特征')
    plt.ylabel('重要性（权重绝对值）')
    plt.title('特征重要性排序')
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print("特征重要性图已保存: results/figures/feature_importance.png")

def analyze_residuals(y_true, y_pred):
    """分析残差"""
    residuals = y_true - y_pred
    
    print("\n" + "="*50)
    print("残差分析")
    print("="*50)
    print(f"残差均值: {np.mean(residuals):.2f}")
    print(f"残差标准差: {np.std(residuals):.2f}")
    print(f"最大正残差: {np.max(residuals):.2f}")
    print(f"最大负残差: {np.min(residuals):.2f}")
    
    # 绘制残差图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差vs预测值')
    
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title('残差分布')
    
    plt.subplot(1, 3, 3)
    # 手动实现正态性检验（替代Q-Q图）
    sorted_residuals = np.sort(residuals.flatten())
    theoretical_quantiles = np.sort(np.random.normal(0, 1, len(sorted_residuals)))
    plt.scatter(theoretical_quantiles, sorted_residuals, alpha=0.5)
    plt.plot([theoretical_quantiles.min(), theoretical_quantiles.max()], 
             [theoretical_quantiles.min(), theoretical_quantiles.max()], 'r--')
    plt.xlabel('理论分位数')
    plt.ylabel('样本分位数')
    plt.title('正态性检验（替代Q-Q图）')
    
    plt.tight_layout()
    plt.savefig('results/figures/residual_analysis.png', dpi=300, bbox_inches='tight')
    print("残差分析图已保存: results/figures/residual_analysis.png")

def calculate_detailed_metrics(y_true, y_pred):
    """计算详细的评估指标"""
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, rmse, mae, mape, r2

def main():
    # 确保结果目录存在
    os.makedirs('results/figures', exist_ok=True)
    
    print("="*60)
    print("医疗保险费用预测模型 - 详细分析报告")
    print("="*60)
    
    # 训练模型
    print("\n1. 模型训练中...")
    model, X_train, X_test, y_train, y_test, mean, std = train_model('data/raw/insurance.csv')
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算详细指标
    print("\n2. 计算评估指标...")
    mse, rmse, mae, mape, r2 = calculate_detailed_metrics(y_test, y_pred)
    
    # 打印评估结果
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    print(f"R²决定系数: {r2:.4f} (解释{r2*100:.1f}%的费用变异)")
    print(f"均方根误差: {rmse:.2f}元")
    print(f"平均绝对误差: {mae:.2f}元")
    print(f"平均绝对百分比误差: {mape:.2f}%")
    print(f"均方误差: {mse:.2f}")
    
    # 特征名称
    feature_names = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 
                    'region_northwest', 'region_southeast', 'region_southwest']
    
    # 特征重要性分析
    analyze_feature_importance(model, feature_names)
    
    # 残差分析
    analyze_residuals(y_test, y_pred)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.cost_history)), model.cost_history)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值（MSE）')
    plt.title('损失函数收敛曲线')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/loss_curve.png', dpi=300, bbox_inches='tight')
    print("\n损失曲线已保存: results/figures/loss_curve.png")
    
    # 绘制实际值vs预测值
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='完美预测线')
    plt.xlabel('实际医疗费用（元）', fontsize=12)
    plt.ylabel('预测医疗费用（元）', fontsize=12)
    plt.title('实际值 vs 预测值散点图', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("实际vs预测图已保存: results/figures/actual_vs_predicted.png")
    
    # 保存详细的预测结果
    predictions = pd.DataFrame({
        '实际费用': y_test.flatten(),
        '预测费用': y_pred.flatten(),
        '残差': (y_test - y_pred).flatten(),
        '相对误差(%)': (np.abs(y_test - y_pred) / y_test * 100).flatten()
    })
    predictions.to_csv('results/detailed_predictions.csv', index=False, encoding='utf-8-sig')
    
    # 保存模型摘要
    summary = pd.DataFrame({
        '指标': ['R²', 'RMSE', 'MAE', 'MAPE', 'MSE'],
        '数值': [r2, rmse, mae, mape, mse],
        '单位': ['', '元', '元', '%', '']
    })
    summary.to_csv('results/model_summary.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print("分析完成！所有结果已保存到results目录")
    print("="*60)
    print("\n生成的文件:")
    print("- results/figures/loss_curve.png (损失曲线)")
    print("- results/figures/actual_vs_predicted.png (预测图)")
    print("- results/figures/feature_importance.png (特征重要性)")
    print("- results/figures/residual_analysis.png (残差分析)")
    print("- results/detailed_predictions.csv (详细预测结果)")
    print("- results/model_summary.csv (模型摘要)")

if __name__ == "__main__":
    main()