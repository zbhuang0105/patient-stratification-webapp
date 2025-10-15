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

# --- 特征显示名称的映射 ---
feature_display_names = {
    'MS': 'Muscle strength',
    'IVDH': 'Mean intervertebral disc height',
    'IVDHR': 'Intervertebral disc height ratio',
    'SS': 'Sacral slope',
    'SA': 'slip angle',
    'LL': 'lumbar lordosis',
    'FJA_R': 'Facet joint angles_Right',
    'FJA_L': 'Facet joint angles_Left',
    'FJA_ABS': 'Left-Right Facet Joint Angle Difference',
    'IVDD': 'Disc degeneration',
    'IVDS': 'lumbar stenosis graded',
    'D_SA': 'dynamic slip angle',
    'D_SPD': 'dynamic slip displacement',
    'IDH': 'lumbar disc herniation',
}


# --- 用户输入界面 ---
st.header("Enter Patient Data")
st.write("Use the fields below to enter the new patient's information.")

gender_map = {'Male': 1, 'Female': 0}
yes_no_map = {'Yes': 1, 'No': 0}

idh_options = [
    "No herniation", 
    "Herniation at the slipped segment", 
    "Bulging at the slipped segment", 
    "Herniation/bulging at non-slipped segments"
]
idh_map = {option: i for i, option in enumerate(idh_options)}

ivdd_options = ['I', 'II', 'III', 'IV', 'V']
ivdd_map = {option: i + 1 for i, option in enumerate(ivdd_options)}

# --- 关键改动 1: 为 lumbar stenosis graded (IVDS) 创建选项和值映射 ---
ivds_options = ['A1', 'A2', 'A3', 'A4', 'B', 'C', 'D']
# 对应值为 1 到 7
ivds_map = {option: i + 1 for i, option in enumerate(ivds_options)}


input_data = {}
cols = st.columns(3)

# --- 动态生成输入框 ---
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        display_label = feature_display_names.get(feature, feature)
        default_value = X_df[feature].mean()

        if feature == 'Gender':
            default_gender_val = round(default_value)
            default_gender_key = [k for k, v in gender_map.items() if v == default_gender_val][0]
            selected_gender = st.selectbox(label=display_label, options=list(gender_map.keys()), index=list(gender_map.keys()).index(default_gender_key), key=feature)
            input_data[feature] = gender_map[selected_gender]
        
        elif feature == 'Age':
            input_data[feature] = st.number_input(
                label=display_label, 
                value=int(round(default_value)),
                min_value=0, max_value=120, step=1, key=feature
            )

        elif feature == 'MS':
            ms_options = [0, 1, 2, 3, 4, 5]
            default_ms = int(round(default_value))
            if default_ms not in ms_options: default_ms = ms_options[0]
            input_data[feature] = st.selectbox(label=display_label, options=ms_options, index=ms_options.index(default_ms), key=feature)
        
        elif feature == 'VAS':
            vas_options = list(range(11))
            default_vas = int(round(default_value))
            if default_vas not in vas_options: default_vas = vas_options[0]
            input_data[feature] = st.selectbox(label=display_label, options=vas_options, index=vas_options.index(default_vas), key=feature)

        elif feature == 'IDH':
            default_idh_val = int(round(default_value))
            if default_idh_val not in range(len(idh_options)): default_idh_val = 0
            selected_idh_text = st.selectbox(
                label=display_label,
                options=idh_options,
                index=default_idh_val,
                key=feature
            )
            input_data[feature] = idh_map[selected_idh_text]

        elif feature == 'IVDD':
            default_ivdd_val = int(round(default_value))
            if default_ivdd_val not in ivdd_map.values(): default_ivdd_val = 1
            default_ivdd_key = [k for k, v in ivdd_map.items() if v == default_ivdd_val][0]
            selected_ivdd_text = st.selectbox(
                label=display_label,
                options=ivdd_options,
                index=ivdd_options.index(default_ivdd_key),
                key=feature
            )
            input_data[feature] = ivdd_map[selected_ivdd_text]

        # --- 关键改动 2: 为 IVDS 添加新的下拉菜单逻辑 ---
        elif feature == 'IVDS':
            default_ivds_val = int(round(default_value))
            if default_ivds_val not in ivds_map.values(): default_ivds_val = 1 # 默认 A1
            default_ivds_key = [k for k, v in ivds_map.items() if v == default_ivds_val][0]
            selected_ivds_text = st.selectbox(
                label=display_label,
                options=ivds_options,
                index=ivds_options.index(default_ivds_key),
                key=feature
            )
            input_data[feature] = ivds_map[selected_ivds_text]

        elif feature in ['OP', 'Smoking', 'Diabetes', 'Hypertension', 'CHD', 'OA']:
            default_yn_val = round(default_value)
            default_yn_key = [k for k, v in yes_no_map.items() if v == default_yn_val][0]
            selected_yn = st.selectbox(label=display_label, options=list(yes_no_map.keys()), index=list(yes_no_map.keys()).index(default_yn_key), key=feature)
            input_data[feature] = yes_no_map[selected_yn]
        else:
            input_data[feature] = st.number_input(label=display_label, value=float(default_value), format="%.2f", key=feature)

# --- 预测按钮和结果展示 (无变化) ---
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
