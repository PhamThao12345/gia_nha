import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import lightgbm as lgb

# Cấu hình trang web Streamlit
st.set_page_config(page_title="Dự báo Giá Nhà - FinTech", layout="wide")
st.title("🏠 Ứng dụng Dự báo Giá nhà (House Prices Prediction)")
st.markdown("Dự án sử dụng các mô hình học máy: Ridge, Lasso, ElasticNet và LightGBM.")

# ===================== Load data =====================
@st.cache_data
def load_data():
    # Lưu ý: Bạn cần up file train.csv và test.csv lên cùng thư mục trên GitHub
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    return train, test

try:
    train, test = load_data()
    st.success(f"Đã tải dữ liệu thành công! Train: {train.shape}, Test: {test.shape}")
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file train.csv hoặc test.csv. Hãy đảm bảo bạn đã upload chúng lên GitHub.")
    st.stop()

# ===================== Quick plots =====================
st.header("📊 Phân tích dữ liệu (EDA)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Phân phối giá nhà (Gốc)")
    fig1, ax1 = plt.subplots()
    sns.histplot(train["SalePrice"], kde=True, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("Phân phối giá nhà (Sau khi Log1p)")
    y = np.log1p(train["SalePrice"])
    fig2, ax2 = plt.subplots()
    sns.histplot(y, kde=True, ax=ax2, color="orange")
    st.pyplot(fig2)

# ===================== Preprocess =====================
with st.spinner("Đang tiền xử lý dữ liệu..."):
    X = train.drop("SalePrice", axis=1)
    full = pd.concat([X, test], axis=0)

    none_cols = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
                 "GarageType","GarageFinish","GarageQual","GarageCond",
                 "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]
    zero_cols = ["GarageYrBlt","GarageArea","GarageCars","BsmtFinSF1","BsmtFinSF2",
                 "BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath"]

    full[none_cols] = full[none_cols].fillna("None")
    full[zero_cols] = full[zero_cols].fillna(0)
    full.fillna(full.median(numeric_only=True), inplace=True)

    numeric_feats = full.select_dtypes(include=[np.number]).columns
    skewed_feats = [c for c in numeric_feats if abs(full[c].skew()) > 0.75]
    for c in skewed_feats:
        full[c] = np.log1p(full[c])

    full = pd.get_dummies(full)
    X_train = full.iloc[:len(train)]
    X_test  = full.iloc[len(train):]

# ===================== Models & Evaluation =====================
st.header("🤖 Huấn luyện và Đánh giá Mô hình")

models = {
    "Ridge": make_pipeline(RobustScaler(), Ridge(alpha=10)),
    "Lasso": make_pipeline(RobustScaler(), Lasso(alpha=0.0003, max_iter=10000)),
    "ElasticNet": make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.85, max_iter=10000)),
}

lgb_model = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=31,
    learning_rate=0.01,
    n_estimators=1000, # Giảm xuống 1000 để app chạy nhanh hơn trên web
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42)

def rmse_cv(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()

if st.button("Bắt đầu huấn luyện (Train Models)"):
    results = {}
    for name, m in models.items():
        score = rmse_cv(m)
        results[name] = score
        st.write(f"✅ {name} RMSE: **{score:.4f}**")
    
    # Fit & Predict
    for m in models.values():
        m.fit(X_train, y)
    lgb_model.fit(X_train, y)
    
    pred = (0.20 * models["Ridge"].predict(X_test) +
            0.25 * models["Lasso"].predict(X_test) +
            0.25 * models["ElasticNet"].predict(X_test) +
            0.30 * lgb_model.predict(X_test))
    
    pred = np.expm1(pred)
    
    st.success("Huấn luyện hoàn tất!")
    
    # Hiển thị kết quả dự báo mẫu
    st.subheader("Kết quả dự báo (5 căn nhà đầu tiên)")
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": pred})
    st.dataframe(submission.head())
    
    # Nút tải file về
    csv = submission.to_csv(index=False).encode('utf-8')
    st.download_button("Tải file submission.csv", data=csv, file_name="submission.csv", mime="text/csv")
