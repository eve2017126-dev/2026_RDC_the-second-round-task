import os
from src.train import train_model
from src.evaluate import evaluate_model
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # 确保结果目录存在
    os.makedirs('results/figures', exist_ok=True)
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, mean, std = train_model('data/raw/insurance.csv')
    
    # 评估模型
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)
    
    # 打印评估结果
    print("Model Evaluation Results:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(model.n_iterations), model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('Loss Function Convergence')
    plt.savefig('results/figures/loss_curve.png')
    print("Loss curve saved to results/figures/loss_curve.png")
    
    # 绘制实际值vs预测值
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Actual vs Predicted Medical Charges')
    plt.savefig('results/figures/actual_vs_predicted.png')
    print("Actual vs predicted plot saved to results/figures/actual_vs_predicted.png")
    
    # 保存预测结果
    predictions = pd.DataFrame({
        'Actual': y_test.flatten(),
        'Predicted': y_pred.flatten()
    })
    predictions.to_csv('results/predictions.csv', index=False)
    print("Predictions saved to results/predictions.csv")

if __name__ == "__main__":
    main()