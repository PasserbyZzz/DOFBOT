import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
import os, joblib
from pathlib import Path

# 1. 读入你的 csv
raw_path = Path('../dataset/60000/dofbot_fk_60000_raw.csv')   # 确保列名与问题一致

# 2. 定义分块列名
cols_q   = ['q1','q2','q3','q4','q5']
cols_xyz = ['x','y','z']
cols_abcd= ['a','b','c','d']
cols_rpy = ['roll','pitch','yaw']

# 3. 自定义周期角度归一化（可选）
class AngleMaxAbsScaler(BaseEstimator, TransformerMixin):
    """先 unwrap 再 MaxAbs，保证角度连续性"""
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.offset_ = np.nanmean(X)          # 可选：去均值
        Xu = np.unwrap(X, axis=0) - self.offset_
        self.scale_ = np.abs(Xu).max(axis=0)
        self.scale_[self.scale_==0] = 1
        return self
    def transform(self, X):
        Xu = np.unwrap(np.asarray(X), axis=0) - self.offset_
        return Xu / self.scale_
    def inverse_transform(self, X):
        return (np.asarray(X) * self.scale_) + self.offset_

# 4. 按块选择 scaler（可自由替换）
scaler_dict = {
    'q'  : MaxAbsScaler(),          # 关节角
    'xyz': StandardScaler(),        # 位置
    'abcd': MaxAbsScaler(),         # 四元数
    'rpy': AngleMaxAbsScaler()      # 欧拉角
}

# 5. 归一化
def normalize(raw_path, scaler_dict, fit=True):
    df = pd.read_csv(raw_path)
    df_norm = df.copy()
    for key, cols in [('q',cols_q), ('xyz',cols_xyz),
                      ('abcd',cols_abcd), ('rpy',cols_rpy)]:
        scaler = scaler_dict[key]
        if fit:
            df_norm[cols] = scaler.fit_transform(df[cols])
        else:
            df_norm[cols] = scaler.transform(df[cols])
    norm_path = raw_path.with_name(raw_path.stem.replace('_raw', '_norm_0929') + '.csv')
    df_norm.to_csv(norm_path, index=False, float_format='%.9f')
    joblib.dump(scaler_dict, raw_path.with_name('scaler_dict.pkl'))
    print(f'已保存至 {norm_path}')
    return df_norm

# 6. 反归一化
def denormalize(df_norm, scaler_dict):
    df_rev = df_norm.copy()
    for key, cols in [('q',cols_q), ('xyz',cols_xyz),
                      ('abcd',cols_abcd), ('rpy',cols_rpy)]:
        scaler = scaler_dict[key]
        df_rev[cols] = scaler.inverse_transform(df_norm[cols])
    return df_rev

# 7. 使用示例
df_train_norm = normalize(raw_path, scaler_dict, fit=True)   # 训练集