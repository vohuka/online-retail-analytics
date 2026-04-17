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

# Gắn nhãn số liệu cho từng điểm trên tập test và dự báo
for x, y in zip(test['ds'], test['y']):
    ax.annotate(f'{y:,.0f}',
                (x, y),
                textcoords='offset points',
                xytext=(0, 5),
                ha='center',
                fontsize=7,
                color='#2E7D32')

for x, yhat in zip(forecast_test['ds'], forecast_test['yhat']):
    ax.annotate(f'{yhat:,.0f}',
                (x, yhat),
                textcoords='offset points',
                xytext=(0, -10),
                ha='center',
                fontsize=7,
                color='#F44336')

ax.legend()
plt.tight_layout()
plt.savefig('outputs/forecast_prophet.png', dpi=150)

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

# ============================================================================
# CÁCH 2: Train/Test chỉ trên tháng 1–10 (loại bỏ tháng 11-12)
# ----------------------------------------------------------------------------
# Lý do: Tháng 11 có Black Friday, tháng 12 có Giáng sinh → doanh thu tăng
# đột biến (spike) khiến model khó dự báo chính xác.
# Thêm vào đó, tháng 12 chỉ có dữ liệu đến 2011-12-09 (không đủ tháng).
# → Thử train/test chỉ trên dữ liệu "bình thường" (tháng 1-10) để đánh giá
#   khả năng dự báo thực sự của model, không bị nhiễu bởi mùa lễ hội.
# Split: Train tháng 1–8, Test tháng 9–10
# ============================================================================
print("\n" + "="*60)
print("CÁCH 2: TRAIN/TEST CHỈ TRÊN THÁNG 1–10")
print("(Loại bỏ tháng 11-12 do Black Friday + Giáng sinh + dữ liệu thiếu)")
print("="*60)

# Chỉ lấy dữ liệu đến hết tháng 10
daily_revenue_10m = daily_revenue[daily_revenue['ds'] < '2011-11-01'].copy()

# Train: tháng 1–8, Test: tháng 9–10
train_v2 = daily_revenue_10m[daily_revenue_10m['ds'] < '2011-09-01']
test_v2  = daily_revenue_10m[daily_revenue_10m['ds'] >= '2011-09-01']

print(f"Train: {len(train_v2)} ngày ({train_v2['ds'].min().date()} → {train_v2['ds'].max().date()})")
print(f"Test:  {len(test_v2)} ngày  ({test_v2['ds'].min().date()} → {test_v2['ds'].max().date()})")

# --- Prophet (Cách 2) ---
model_v2 = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
model_v2.fit(train_v2)

future_v2 = model_v2.make_future_dataframe(periods=len(test_v2))
forecast_v2 = model_v2.predict(future_v2)

forecast_test_v2 = forecast_v2[forecast_v2['ds'].isin(test_v2['ds'])][['ds','yhat','yhat_lower','yhat_upper']]
forecast_test_v2 = forecast_test_v2.merge(test_v2, on='ds')

mae_v2  = mean_absolute_error(forecast_test_v2['y'], forecast_test_v2['yhat'])
rmse_v2 = np.sqrt(mean_squared_error(forecast_test_v2['y'], forecast_test_v2['yhat']))
mape_v2 = (abs((forecast_test_v2['y'] - forecast_test_v2['yhat']) / forecast_test_v2['y']).mean()) * 100

print("\n=== KẾT QUẢ DỰ BÁO — Prophet (Cách 2: tháng 1–10) ===")
print(f"MAE:  £{mae_v2:,.0f}")
print(f"RMSE: £{rmse_v2:,.0f}")
print(f"MAPE: {mape_v2:.1f}%")

# --- XGBoost (Cách 2) ---
data_feat_10m = make_time_features(daily_revenue_10m)
data_feat_10m = data_feat_10m.dropna()

train_feat_v2 = data_feat_10m[data_feat_10m['ds'] < '2011-09-01']
test_feat_v2  = data_feat_10m[data_feat_10m['ds'] >= '2011-09-01']

X_train_v2, y_train_v2 = train_feat_v2[feature_cols], train_feat_v2['y']
X_test_v2,  y_test_v2  = test_feat_v2[feature_cols],  test_feat_v2['y']

xgb_v2 = XGBRegressor(n_estimators=200, learning_rate=0.05,
                       max_depth=5, random_state=42, verbosity=0)
xgb_v2.fit(X_train_v2, y_train_v2)
y_pred_xgb_v2 = xgb_v2.predict(X_test_v2)

mae_xgb_v2  = mean_absolute_error(y_test_v2, y_pred_xgb_v2)
rmse_xgb_v2 = np.sqrt(mean_squared_error(y_test_v2, y_pred_xgb_v2))
mape_xgb_v2 = (abs((y_test_v2 - y_pred_xgb_v2) / y_test_v2).mean()) * 100

print("\n=== KẾT QUẢ XGBoost (Cách 2: tháng 1–10) ===")
print(f"MAE:  £{mae_xgb_v2:,.0f}")
print(f"RMSE: £{rmse_xgb_v2:,.0f}")
print(f"MAPE: {mape_xgb_v2:.1f}%")

# Bảng so sánh Cách 2
comparison_v2 = pd.DataFrame({
    'Model':  ['Prophet', 'XGBoost'],
    'MAE (£)': [f'{mae_v2:.0f}',     f'{mae_xgb_v2:.0f}'],
    'RMSE (£)': [f'{rmse_v2:.0f}',    f'{rmse_xgb_v2:.0f}'],
    'MAPE (%)': [f'{mape_v2:.1f}%',   f'{mape_xgb_v2:.1f}%']
})
print("\n=== BẢNG SO SÁNH MODEL — Cách 2 (tháng 1–10) ===")
print(comparison_v2.to_string(index=False))

# So sánh cùng 1 thuật toán giữa Cách 1 và Cách 2
# 1) Prophet: chọn phiên bản có MAPE tốt hơn để export
if mape <= mape_v2:
    best_prophet_model = model
    best_prophet_approach = 'Cách 1 (test T11-12)'
    best_prophet_mape = mape
else:
    best_prophet_model = model_v2
    best_prophet_approach = 'Cách 2 (test T9-10)'
    best_prophet_mape = mape_v2

# 2) XGBoost: chọn phiên bản có MAPE tốt hơn để export
if mape_xgb <= mape_xgb_v2:
    best_xgb_model = xgb
    best_xgb_approach = 'Cách 1 (test T11-12)'
    best_xgb_mape = mape_xgb
else:
    best_xgb_model = xgb_v2
    best_xgb_approach = 'Cách 2 (test T9-10)'
    best_xgb_mape = mape_xgb_v2

print("\n=== SO SÁNH THEO TỪNG THUẬT TOÁN (C1 vs C2) ===")
print(f"✅ Prophet tốt hơn ở: {best_prophet_approach} (MAPE: {best_prophet_mape:.1f}%)")
print(f"✅ XGBoost tốt hơn ở: {best_xgb_approach} (MAPE: {best_xgb_mape:.1f}%)")

# So sánh tổng hợp giữa Cách 1 (gồm tháng 11-12) và Cách 2 (chỉ tháng 1-10)
print("\n" + "="*60)
print("SO SÁNH TỔNG HỢP: Cách 1 (test tháng 11-12) vs Cách 2 (test tháng 9-10)")
print("="*60)
comparison_all = pd.DataFrame({
    'Cách / Model': [
        'C1: Prophet (test T11-12)',
        'C1: XGBoost (test T11-12)',
        'C2: Prophet (test T9-10)',
        'C2: XGBoost (test T9-10)'
    ],
    'MAE (£)':  [f'{mae:.0f}', f'{mae_xgb:.0f}', f'{mae_v2:.0f}', f'{mae_xgb_v2:.0f}'],
    'RMSE (£)': [f'{rmse:.0f}', f'{rmse_xgb:.0f}', f'{rmse_v2:.0f}', f'{rmse_xgb_v2:.0f}'],
    'MAPE (%)': [f'{mape:.1f}%', f'{mape_xgb:.1f}%', f'{mape_v2:.1f}%', f'{mape_xgb_v2:.1f}%']
})
print(comparison_all.to_string(index=False))
print("\n→ Nếu MAPE Cách 2 thấp hơn nhiều → khẳng định tháng 11-12 gây nhiễu lớn")
print("→ Cách 2 phản ánh đúng hơn khả năng dự báo của model trên dữ liệu bình thường")

# Vẽ biểu đồ dự báo Cách 2
fig2, ax2 = plt.subplots(figsize=(13, 5))
ax2.plot(train_v2['ds'], train_v2['y'], color='#1565C0', label='Doanh thu thực (train)', linewidth=1)
ax2.plot(test_v2['ds'],  test_v2['y'],  color='#2E7D32', label='Doanh thu thực (test)',  linewidth=2)
ax2.plot(forecast_test_v2['ds'], forecast_test_v2['yhat'],
         color='#F44336', label='Dự báo (Prophet)', linewidth=2, linestyle='--')
ax2.fill_between(forecast_test_v2['ds'],
                 forecast_test_v2['yhat_lower'], forecast_test_v2['yhat_upper'],
                 alpha=0.15, color='#F44336', label='Khoảng tin cậy')
ax2.set_title('Dự báo doanh thu — Prophet (Cách 2: chỉ tháng 1–10, tránh mùa lễ hội)', fontsize=13)
ax2.set_xlabel('Ngày')
ax2.set_ylabel('Doanh thu (£)')

for x, y_val in zip(test_v2['ds'], test_v2['y']):
    ax2.annotate(f'{y_val:,.0f}',
                 (x, y_val),
                 textcoords='offset points',
                 xytext=(0, 5),
                 ha='center',
                 fontsize=7,
                 color='#2E7D32')

for x, yhat_val in zip(forecast_test_v2['ds'], forecast_test_v2['yhat']):
    ax2.annotate(f'{yhat_val:,.0f}',
                 (x, yhat_val),
                 textcoords='offset points',
                 xytext=(0, -10),
                 ha='center',
                 fontsize=7,
                 color='#F44336')

ax2.legend()
plt.tight_layout()
plt.savefig('outputs/forecast_prophet_v2.png', dpi=150)


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
plt.savefig('outputs/confusion_matrix.png', dpi=150)
# Ô trên-trái: đúng active, ô dưới-phải: đúng churn
# Ô trên-phải: báo nhầm churn, ô dưới-trái: bỏ sót churn

# Feature Importance: feature nào quan trọng nhất?
feat_imp = pd.Series(rf.feature_importances_, index=features_churn).sort_values(ascending=True)

plt.figure(figsize=(7, 4))
ax = feat_imp.plot(kind='barh', color='#42A5F5')
plt.title('Tầm quan trọng của từng feature', fontsize=13)
plt.xlabel('Importance Score')

# Gắn nhãn số liệu cho từng cột feature importance
for i, v in enumerate(feat_imp.values):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)


# Lưu model
with open('outputs/prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('outputs/churn_rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Lưu model forecast tốt nhất cho từng thuật toán (so sánh C1 vs C2)
with open('outputs/best_prophet_model.pkl', 'wb') as f:
    pickle.dump(best_prophet_model, f)
with open('outputs/best_xgb_model.pkl', 'wb') as f:
    pickle.dump(best_xgb_model, f)

# Lưu thông tin model thắng cuộc theo từng thuật toán để dễ tra cứu
best_forecast_summary = pd.DataFrame({
    'Algorithm': ['Prophet', 'XGBoost'],
    'Best Approach': [best_prophet_approach, best_xgb_approach],
    'Best MAPE (%)': [round(best_prophet_mape, 1), round(best_xgb_mape, 1)],
    'Model File': ['best_prophet_model.pkl', 'best_xgb_model.pkl']
})
best_forecast_summary.to_csv('outputs/best_forecast_models_summary.csv', index=False)

# Lưu dự báo
rfm['ChurnProba'] = rf.predict_proba(rfm[features_churn])[:, 1]
rfm.to_csv('outputs/rfm_with_predictions.csv', index=False)

print("✅ Phase 4 hoàn thành!")
print("   Sales Forecasting: Prophet + XGBoost đã so sánh")
print("   Best Forecast Models: đã export Prophet + XGBoost tốt nhất (C1 vs C2)")
print("   Churn Prediction:  Random Forest đã huấn luyện")


