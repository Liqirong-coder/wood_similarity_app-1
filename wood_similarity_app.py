import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import os

# ====== 页面设置 ======
st.set_page_config(
    page_title="木材纹理相似度评估系统 / Wood Texture Similarity System",
    page_icon="blfu_logo.jpg",  # 相对路径
    layout="centered"
)

# 显示 logo
st.image("blfu_logo.jpg", width=100)

# ====== 多语言字典 ======
lang_dict = {
    "zh": {
        "title": "木材纹理相似度评估系统",
        "desc": "上传原材图像和替代材图像，系统将模拟主观感受对纹理相似度进行评分。",
        "upload_ori": "上传原材图像",
        "upload_imi": "上传替代材图像（可上传多张）",
        "start_btn": "开始评估",
        "warning_upload": "请先上传原材和替代材图像！",
        "results": "🏆 相似度预测结果",
        "likert_col": "里克特量表(1-5)",
        "sim_col": "模拟主观评分（%）",
        "image_col": "替代材图像",
        "index_label": "排名",
        "success_msg": "结果已保存至桌面："
    },
    "en": {
        "title": "Wood Texture Similarity Evaluation System",
        "desc": "Upload the original wood image and replacement images. The system will simulate subjective perception to score texture similarity.",
        "upload_ori": "Upload Original Wood Image",
        "upload_imi": "Upload Replacement Images (Multiple Allowed)",
        "start_btn": "Start Evaluation",
        "warning_upload": "Please upload both original and replacement images first!",
        "results": "🏆 Similarity Evaluation Results",
        "likert_col": "Likert Scale (1-5)",
        "sim_col": "Simulated Subjective Score (%)",
        "image_col": "Replacement Image",
        "index_label": "Rank",
        "success_msg": "Results have been saved to Desktop: "
    }
}

# ====== 语言选择 ======
lang_choice = st.selectbox("🌐 语言 / Language", ["中文", "English"])
lang = "zh" if lang_choice == "中文" else "en"

st.title(lang_dict[lang]["title"])
st.write(lang_dict[lang]["desc"])

# ====== 纹理特征函数 ======
def texture_direction(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    vertical = np.sum(magnitude_spectrum[:, ccol-5:ccol+5])
    horizontal = np.sum(magnitude_spectrum[crow-5:crow+5, :])
    return vertical / (horizontal + vertical + 1e-8)

def texture_clarity(img):
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var / (laplacian_var + 1000)

def texture_contrast(img):
    return img.std() / (np.mean(img) + 1e-8)

def texture_uniformity(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.ravel() / hist.sum()
    return np.sum(hist**2)

def texture_coarseness(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return 1.0 / (np.mean(grad_mag) + 1e-8)

def extract_features(img):
    img = np.array(img.convert("L").resize((256,256)))
    return [
        texture_direction(img),
        texture_clarity(img),
        texture_contrast(img),
        texture_uniformity(img),
        texture_coarseness(img)
    ]

# ====== 加载模型 ======
model_file = "random_forest_model.joblib"  # 相对路径
rf = joblib.load(model_file)

# ====== 上传图像 ======
st.subheader(lang_dict[lang]["upload_ori"])
ori_file = st.file_uploader(lang_dict[lang]["upload_ori"], type=["jpg","png","jpeg"])

st.subheader(lang_dict[lang]["upload_imi"])
imi_files = st.file_uploader(lang_dict[lang]["upload_imi"], type=["jpg","png","jpeg"], accept_multiple_files=True)

# ====== 开始评估 ======
if st.button(lang_dict[lang]["start_btn"]):
    if ori_file is None or len(imi_files) == 0:
        st.warning(lang_dict[lang]["warning_upload"])
    else:
        ori_img = Image.open(ori_file)
        features_ori = extract_features(ori_img)

        results = []
        for imi_file in imi_files:
            imi_img = Image.open(imi_file)
            features_imi = extract_features(imi_img)
            sims = [1 - abs(o - i) / (o + i + 1e-8) for o, i in zip(features_ori, features_imi)]
            features_array = np.array(sims).reshape(1,-1)
            y_pred = rf.predict(features_array)[0]
            y_pred = np.clip(y_pred, 0.0, 1.0) * 100
            results.append({
                lang_dict[lang]["image_col"]: imi_file.name,
                lang_dict[lang]["sim_col"]: y_pred
            })

        # ====== 构建 DataFrame ======
        df = pd.DataFrame(results).sort_values(by=lang_dict[lang]["sim_col"], ascending=False).reset_index(drop=True)
        df.index += 1
        df[lang_dict[lang]["likert_col"]] = (1 + df[lang_dict[lang]["sim_col"]] / 100 * 4).round().astype(int)

        # ====== 网页显示 ======
        st.subheader(lang_dict[lang]["results"])
        st.dataframe(df)

        # ====== 导出 Excel ======
        output_path = os.path.join(os.path.expanduser("~"), "Desktop", "wood_similarity_results.xlsx")
        df.to_excel(output_path, index_label=lang_dict[lang]["index_label"])
        st.success(f"{lang_dict[lang]['success_msg']}{output_path}")
