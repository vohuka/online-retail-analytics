import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns


df_all = pd.read_csv('outputs/df_all_clean.csv', parse_dates=['InvoiceDate'])

# Tổng hợp doanh thu theo ngày
daily_revenue = df_all.groupby(
    df_all['InvoiceDate'].dt.date
)['Revenue'].sum().reset_index()
daily_revenue.columns = ['ds', 'y']  # Prophet yêu cầu tên cột là 'ds' và 'y'
daily_revenue['ds'] = pd.to_datetime(daily_revenue['ds'])

print(f"Dữ liệu ngày: {len(daily_revenue)} ngày")
print(daily_revenue.head())

# Chia train/test: train đến 31/10/2011, test tháng 11-12/2011
train = daily_revenue[daily_revenue['ds'] < '2011-11-01']
test  = daily_revenue[daily_revenue['ds'] >= '2011-11-01']

print(f"Train: {len(train)} ngày ({train['ds'].min().date()} → {train['ds'].max().date()})")
print(f"Test:  {len(test)} ngày  ({test['ds'].min().date()} → {test['ds'].max().date()})")

# Huấn luyện mô hình Prophet
model = Prophet(
    yearly_seasonality=True,   # Có chu kỳ năm (Giáng sinh, v.v.)
    weekly_seasonality=True,   # Có chu kỳ tuần (cuối tuần mua ít hơn)
    daily_seasonality=False,   # Không cần chu kỳ ngày
    seasonality_mode='multiplicative'  # Seasonality nhân (tốt hơn cho bán lẻ)
)

model.fit(train)
print("✅ Huấn luyện xong!")

# Dự báo 60 ngày tiếp theo (tháng 11-12)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Lấy phần dự báo tương ứng với test set
forecast_test = forecast[forecast['ds'].isin(test['ds'])][['ds','yhat','yhat_lower','yhat_upper']]
forecast_test = forecast_test.merge(test, on='ds')

# Tính độ sai số
mae  = mean_absolute_error(forecast_test['y'], forecast_test['yhat'])
rmse = np.sqrt(mean_squared_error(forecast_test['y'], forecast_test['yhat']))
mape = (abs((forecast_test['y'] - forecast_test['yhat']) / forecast_test['y']).mean()) * 100

print("=== KẾT QUẢ DỰ BÁO (Prophet) ===")
print(f"MAE  (Sai số trung bình tuyệt đối):  £{mae:,.0f}")
print(f"RMSE (Căn sai số bình phương TB):     £{rmse:,.0f}")
print(f"MAPE (Sai số phần trăm TB):           {mape:.1f}%")
print("\nGiải thích: MAPE = X% nghĩa là dự báo lệch X% so với thực tế")


# Vẽ biểu đồ dự báo
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train['ds'], train['y'], color='#1565C0', label='Doanh thu thực (train)', linewidth=1)
ax.plot(test['ds'],  test['y'],  color='#2E7D32', label='Doanh thu thực (test)',  linewidth=2)
ax.plot(forecast_test['ds'], forecast_test['yhat'],
        color='#F44336', label='Dự báo (Prophet)', linewidth=2, linestyle='--')
ax.fill_between(forecast_test['ds'],
                forecast_test['yhat_lower'], forecast_test['yhat_upper'],
                alpha=0.15, color='#F44336', label='Khoảng tin cậy')
ax.set_title('Dự báo doanh thu — Prophet', fontsize=14)
ax.set_xlabel('Ngày')
ax.set_ylabel('Doanh thu (£)')
ax.legend()
plt.tight_layout()
plt.savefig('forecast_prophet.png', dpi=150)

# Tạo features từ ngày (time-based features)
def make_time_features(df):
    df = df.copy()
    df['dayofweek'] = df['ds'].dt.dayofweek   # 0=Thứ 2, 6=CN
    df['month']     = df['ds'].dt.month
    df['day']       = df['ds'].dt.day
    df['weekofyear']= df['ds'].dt.isocalendar().week.astype(int)
    # Lag features: doanh thu 7 ngày trước, 14 ngày trước
    df['lag_7']  = df['y'].shift(7)
    df['lag_14'] = df['y'].shift(14)
    df['lag_28'] = df['y'].shift(28)
    # Rolling mean: trung bình 7 ngày gần nhất
    df['rolling_7'] = df['y'].shift(1).rolling(7).mean()
    return df

data_feat = make_time_features(daily_revenue)
data_feat = data_feat.dropna()  # Xóa dòng NaN do lag

feature_cols = ['dayofweek','month','day','weekofyear','lag_7','lag_14','lag_28','rolling_7']

# Chia train/test (cùng ngưỡng như Prophet)
train_feat = data_feat[data_feat['ds'] < '2011-11-01']
test_feat  = data_feat[data_feat['ds'] >= '2011-11-01']

X_train, y_train = train_feat[feature_cols], train_feat['y']
X_test,  y_test  = test_feat[feature_cols],  test_feat['y']

# Train XGBoost
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                   max_depth=5, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mape_xgb = (abs((y_test - y_pred_xgb) / y_test).mean()) * 100

print("=== KẾT QUẢ XGBoost ===")
print(f"MAE:  £{mae_xgb:,.0f}")
print(f"RMSE: £{rmse_xgb:,.0f}")
print(f"MAPE: {mape_xgb:.1f}%")

# Bảng so sánh 2 model
comparison = pd.DataFrame({
    'Model':  ['Prophet', 'XGBoost'],
    'MAE (£)': [f'{mae:.0f}',     f'{mae_xgb:.0f}'],
    'RMSE (£)': [f'{rmse:.0f}',    f'{rmse_xgb:.0f}'],
    'MAPE (%)': [f'{mape:.1f}%',   f'{mape_xgb:.1f}%']
})
print("\n=== BẢNG SO SÁNH MODEL DỰ BÁO ===")
print(comparison.to_string(index=False))
print("\n→ Model có MAPE thấp hơn = dự báo chính xác hơn")



rfm = pd.read_csv('outputs/rfm_segments.csv')

# Tạo nhãn churn: Recency > 90 ngày = đã churn
rfm['Churned'] = (rfm['Recency'] > 90).astype(int)
# 1 = đã churn (lâu không mua), 0 = còn active

print("=== PHÂN PHỐI NHÃN CHURN ===")
churn_dist = rfm['Churned'].value_counts()
print(f"Còn active (0):  {churn_dist[0]:,} khách ({churn_dist[0]/len(rfm)*100:.1f}%)")
print(f"Đã churn   (1):  {churn_dist[1]:,} khách ({churn_dist[1]/len(rfm)*100:.1f}%)")

# Chuẩn bị features và target
# Không dùng Recency trực tiếp vì ta đã dùng nó để tạo nhãn
# (sẽ gây data leakage — "gian lận" vô ý)
features_churn = ['Frequency', 'Monetary', 'Diversity']
X = rfm[features_churn]
y = rfm['Churned']

# Chia 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y: đảm bảo tỉ lệ churn trong train và test bằng nhau

print(f"Train: {len(X_train):,} khách")
print(f"Test:  {len(X_test):,} khách")

# Huấn luyện Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Dự báo
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]  # Xác suất churn (0.0 → 1.0)

# Đánh giá
print("=== KẾT QUẢ CHURN PREDICTION ===")
print(classification_report(y_test, y_pred,
                             target_names=['Còn active','Đã churn']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print("\nGiải thích:")
print("Precision: Trong số khách mô hình dự báo churn, bao nhiêu % đúng")
print("Recall:    Trong số khách thực sự churn, mô hình tìm đúng bao nhiêu %")
print("AUC-ROC:   Điểm tổng hợp (0.5=ngẫu nhiên, 1.0=hoàn hảo)")

# Vẽ Confusion Matrix
# (bảng so sánh dự báo vs thực tế)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Dự báo: Active','Dự báo: Churn'],
            yticklabels=['Thực tế: Active','Thực tế: Churn'])
plt.title('Confusion Matrix — Churn Prediction', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
# Ô trên-trái: đúng active, ô dưới-phải: đúng churn
# Ô trên-phải: báo nhầm churn, ô dưới-trái: bỏ sót churn

# Feature Importance: feature nào quan trọng nhất?
feat_imp = pd.Series(rf.feature_importances_, index=features_churn).sort_values(ascending=True)

plt.figure(figsize=(7, 4))
feat_imp.plot(kind='barh', color='#42A5F5')
plt.title('Tầm quan trọng của từng feature', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)



# Lưu model
with open('outputs/prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('outputs/churn_rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Lưu dự báo
rfm['ChurnProba'] = rf.predict_proba(rfm[features_churn])[:, 1]
rfm.to_csv('outputs/rfm_with_predictions.csv', index=False)

print("✅ Phase 4 hoàn thành!")
print("   Sales Forecasting: Prophet + XGBoost đã so sánh")
print("   Churn Prediction:  Random Forest đã huấn luyện")


