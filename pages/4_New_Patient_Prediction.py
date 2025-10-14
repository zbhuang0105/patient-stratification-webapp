import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 从主应用文件导入初始化函数
try:
    from app import initialize_data
except ImportError:
    st.error("Could not import the main app. Please ensure the file structure is correct.")
    st.stop()

# 在页面加载时运行数据初始化
initialize_data()

st.title("🔬 New Patient Prediction")

# 检查数据是否加载成功
if 'model' not in st.session_state:
    st.error("Data could not be loaded. Please ensure the main app runs correctly and return to the main page.")
    st.stop()

# 从 session_state 获取加载好的数据
model = st.session_state['model']
subcluster_model = st.session_state['subcluster_model']
explainer = st.session_state['explainer']
X_df = st.session_state['X_df']
feature_names = st.session_state['feature_names']
class_names = st.session_state['class_names']

# 用户输入界面
st.header("Enter Patient Data")
st.write("Use the fields below to enter the new patient's information.")

gender_map = {'Male': 1, 'Female': 0}
yes_no_map = {'Yes': 1, 'No': 0}

input_data = {}
cols = st.columns(3)

# 动态生成输入框
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        default_value = X_df[feature].mean()
        if feature == 'Gender':
            default_gender_val = round(default_value)
            default_gender_key = [k for k, v in gender_map.items() if v == default_gender_val][0]
            selected_gender = st.selectbox(label=feature, options=list(gender_map.keys()), index=list(gender_map.keys()).index(default_gender_key), key=feature)
            input_data[feature] = gender_map[selected_gender]
        
        # --- 关键改动：为 'Age' 添加特定处理 ---
        elif feature == 'Age':
            input_data[feature] = st.number_input(
                label=feature, 
                value=int(round(default_value)), # 将默认值转为整数
                min_value=0,                      # 设置合理的最小值
                max_value=120,                    # 设置合理的最小值
                step=1,                           # 步长设为1
                key=feature
            )

        elif feature == 'MS':
            ms_options = [0, 1, 2, 3, 4, 5]
            default_ms = int(round(default_value))
            if default_ms not in ms_options: default_ms = ms_options[0]
            input_data[feature] = st.selectbox(label=feature, options=ms_options, index=ms_options.index(default_ms), key=feature)
        elif feature == 'VAS':
            input_data[feature] = st.slider(label=feature, min_value=0, max_value=10, value=int(round(default_value)), step=1, key=feature)
        elif feature in ['OP', 'Smoking', 'Diabetes', 'Hypertension', 'CHD', 'OA']:
            default_yn_val = round(default_value)
            default_yn_key = [k for k, v in yes_no_map.items() if v == default_yn_val][0]
            selected_yn = st.selectbox(label=feature, options=list(yes_no_map.keys()), index=list(yes_no_map.keys()).index(default_yn_key), key=feature)
            input_data[feature] = yes_no_map[selected_yn]
        else:
            # 其他所有数值型特征保持不变
            input_data[feature] = st.number_input(label=feature, value=float(default_value), format="%.2f", key=feature)

# 预测按钮和结果展示
if st.button("Get Prediction", type="primary"):
    st.header("Prediction Results")

    patient_df = pd.DataFrame([input_data], columns=feature_names)

    prediction_idx = model.predict(patient_df)[0]
    prediction_label = class_names[prediction_idx]
    prediction_proba = model.predict_proba(patient_df)

    final_prediction_label = prediction_label
    if prediction_label == 'Moderate': 
        subcluster_prediction_idx = subcluster_model.predict(patient_df)[0]
        subcluster_label = "A" if subcluster_prediction_idx == 0 else "B"
        final_prediction_label = f"Moderate_{subcluster_label}"

    col1, col2 = st.columns([1, 2])
    col1.metric("Predicted Severity", final_prediction_label)
    
    proba_df = pd.DataFrame(prediction_proba, columns=class_names, index=["Probability"]).T
    col2.dataframe(proba_df.style.format("{:.2%}"))

    st.header("Prediction Explanation (SHAP Waterfall Plots)")
    st.write("The charts below explain why the model made this prediction. Red arrows push the prediction towards a class, while blue arrows push it away.")

    shap_values_patient = explainer(patient_df)

    for i, class_name in enumerate(class_names):
        st.subheader(f"Explanation for '{class_name}' Prediction")
        
        explanation = shap.Explanation(
            values=shap_values_patient.values[0, :, i],
            base_values=shap_values_patient.base_values[0, i],
            data=patient_df.iloc[0],
            feature_names=feature_names
        )
        
        try:
            fig, ax = plt.subplots()
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not generate SHAP plot for class '{class_name}': {e}")
