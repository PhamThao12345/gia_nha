import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet

import lightgbm as lgb


# ===================== Load data =====================
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test  = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(train.shape, test.shape)


# ===================== Quick plots =====================
def plot_hist(data, title):
    plt.figure(figsize=(8, 4))
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.show()

plot_hist(train["SalePrice"], "SalePrice (original)")

y = np.log1p(train["SalePrice"])
plot_hist(y, "SalePrice (log1p)")


# ===================== Preprocess =====================
X = train.drop("SalePrice", axis=1)
full = pd.concat([X, test], axis=0)

none_cols = [
    "PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
    "GarageType","GarageFinish","GarageQual","GarageCond",
    "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"
]
zero_cols = [
    "GarageYrBlt","GarageArea","GarageCars",
    "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
    "BsmtFullBath","BsmtHalfBath"
]

# fill NA (same behavior)
full[none_cols] = full[none_cols].fillna("None")
full[zero_cols] = full[zero_cols].fillna(0)

# numeric median fill (same behavior)
full.fillna(full.median(numeric_only=True), inplace=True)

# log1p skewed numeric features (same threshold)
numeric_feats = full.select_dtypes(include=[np.number]).columns
skewed_feats = [c for c in numeric_feats if abs(full[c].skew()) > 0.75]
for c in skewed_feats:
    full[c] = np.log1p(full[c])

# one-hot
full = pd.get_dummies(full)

X_train = full.iloc[:len(train)]
X_test  = full.iloc[len(train):]


# ===================== Models =====================
models = {
    "Ridge": make_pipeline(RobustScaler(), Ridge(alpha=10)),
    "Lasso": make_pipeline(RobustScaler(), Lasso(alpha=0.0003, max_iter=10000)),
    "ElasticNet": make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.85, max_iter=10000)),
}

lgb_model = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=31,
    learning_rate=0.01,
    n_estimators=3000,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

def rmse_cv(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(
        model, X_train, y,
        scoring="neg_mean_squared_error",
        cv=kf
    ))
    return rmse.mean()

for name, m in models.items():
    print(f"{name} RMSE:", rmse_cv(m))


# ===================== Fit & Predict =====================
for m in models.values():
    m.fit(X_train, y)
lgb_model.fit(X_train, y)

pred = (
    0.20 * models["Ridge"].predict(X_test) +
    0.25 * models["Lasso"].predict(X_test) +
    0.25 * models["ElasticNet"].predict(X_test) +
    0.30 * lgb_model.predict(X_test)
)

pred = np.expm1(pred)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": pred
})

submission.to_csv("submission.csv", index=False)
submission.head()
