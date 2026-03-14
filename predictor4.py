# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib

# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt

# 加载训练好的模型
model = joblib.load('GNB.pkl')

# 从 X_test.csv 文件加载测试数据，以便用于 LIME 解释器
X_train_resampled = pd.read_csv('X_train_resampled.csv')

# 定义特征名，对应数据集中的列名
feature_names = [
    "EDV",      # 舒张期末容积
    "EF",       # 射血分数
    "FWLS",      # 游离壁的纵向应变 
    "SLS"        # 间隔的纵向应变
]

# Streamlit 用户界面
st.title("Predictor of Severe Pulmonary Hypertension (2026 Beta)")  # 设置网页标题

# 数值输入框
EDV = st.number_input("End-diastolic volume (EDV), ml:", min_value=0.00, max_value=500.00, step=0.01, value=0.00)
EF = st.number_input("Ejection fraction (EF), %:", min_value=0.00, max_value=100.00, step=0.01, value=0.00)
FWLS = st.number_input("Free wall strain (FWLS), %:", min_value=-100.00, max_value=0.00, step=0.01, value=0.00)
SLS = st.number_input("Septal strain (SLS), %:", min_value=-100.00, max_value=0.00, step=0.01, value=0.00)


# 处理输入数据并进行预测
feature_values = [EDV, EF, FWLS, SLS]  # 将用户输入的特征存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0：低风险，1：高风险）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: High Risk, 0: Low Risk)")
    st.write(f"**Prediction Probabilities:** {predicted_proba[predicted_class]}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为 1（高风险）
    if predicted_class == 1:
        advice = (            
            f"Severe pulmonary hypertension refers to a mean pulmonary arterial pressure greater than 50 mmHg."
            f"According to the model, this subject has a high risk of severe pulmonary hypertension."        
            f"The model predicts a probability of {probability:.1f}%."
            "It's advised to consult with healthcare providers for further evaluation and possible intervention."
        )
     # 如果预测类别为 0（低风险）
    else:
        advice = (
            f"Severe pulmonary hypertension refers to a mean pulmonary arterial pressure greater than 50 mmHg."
            f"According to the model, this subject has a low risk of severe pulmonary hypertension."  
            f"The model predicts a probability of {probability:.1f}%."
            "Please continue regular check-ups with healthcare providers."
        )
    # 显示建议
    st.write(advice)    
    
    
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    # 创建 SHAP 解释器  
    def model_predict_proba(data_to_predict_np):
        data_df = pd.DataFrame(data_to_predict_np, columns=X_train_resampled.columns)
        proba = model.predict_proba(data_df)
        return proba
    
    explainer_shap = shap.KernelExplainer(model_predict_proba, X_train_resampled)
    
    
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 根据预测类别显示 SHAP图
    # 期望值（基线值）
    # 解释类别 1（患病）的 SHAP 值
    # 特征值数据
    # 使用Matplotlib绘图
    if predicted_class == 1:    
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用Matplotlib绘图
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption="SHAP Force Plot Explanation")