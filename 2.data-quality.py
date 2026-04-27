import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ghi CSV sạch ra đĩa (KHÔNG ghi đè online_retail.csv — file gốc chỉ đọc)
os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('online_retail.csv')
print(f"✅ Đọc xong! Dataset có {df.shape[0]:,} hàng, {df.shape[1]} cột")
df.head(3)

print("=== TRƯỚC KHI LÀM SẠCH ===")
print(f"Tổng số dòng: {len(df):,}")

# --- Loại 1: Hóa đơn hủy (InvoiceNo bắt đầu bằng 'C') ---
# Giải thích: 'C536379' là đơn hủy tương ứng với đơn '536379'
# Dữ liệu này không phản ánh hành vi mua hàng thực → loại bỏ
cancelled_mask = df['InvoiceNo'].astype(str).str.startswith('C')
print(f"\nHóa đơn hủy (bắt đầu 'C'): {cancelled_mask.sum():,} dòng → SẼ XÓA")

# --- Loại 2: Quantity âm (hàng trả lại) ---
# Giải thích: Quantity = -3 nghĩa là khách trả lại 3 sản phẩm
# Những dòng này làm lệch phân tích doanh thu → loại bỏ
neg_qty_mask = df['Quantity'] <= 0
print(f"Quantity âm hoặc = 0:       {neg_qty_mask.sum():,} dòng → SẼ XÓA")

# --- Loại 3: UnitPrice âm hoặc = 0 ---
# Giải thích: Sản phẩm giá 0 là quà tặng/điều chỉnh kế toán, không phải giao dịch thực
neg_price_mask = df['UnitPrice'] <= 0
print(f"UnitPrice âm hoặc = 0:      {neg_price_mask.sum():,} dòng → SẼ XÓA")

# Xóa tất cả các dòng lỗi trên (dùng ~ nghĩa là "KHÔNG phải")
df_clean = df[~cancelled_mask & ~neg_qty_mask & ~neg_price_mask].copy()

print(f"\n=== SAU KHI XÓA DÒNG LỖI ===")
print(f"Còn lại: {len(df_clean):,} dòng")
print(f"Đã xóa:  {len(df) - len(df_clean):,} dòng ({(len(df)-len(df_clean))/len(df)*100:.1f}%)")

print("=== Ô TRỐNG SAU BƯỚC 2.1 ===")
missing = df_clean.isnull().sum()
for col in missing[missing > 0].index:
    pct = missing[col] / len(df_clean) * 100
    print(f"{col:15s}: {missing[col]:,} ô trống ({pct:.1f}%)")


# Description trống rất ít → xóa luôn, không ảnh hưởng đáng kể
before = len(df_clean)
df_clean = df_clean.dropna(subset=['Description'])
print(f"Xóa {before - len(df_clean)} dòng thiếu Description")
# Giải thích 'dropna': drop = xóa, na = not available (ô trống)
# subset=['Description'] = chỉ xóa dòng nào mà CỘT Description trống

# Giải thích tại sao KHÔNG xóa ngay:
# CustomerID trống = khách vãng lai (guest checkout), không đăng ký tài khoản
# Những giao dịch này vẫn có giá trị cho phân tích doanh thu tổng thể
# → Ta sẽ tạo 2 bản dataset:
#   - df_all: giữ tất cả (dùng cho phân tích doanh thu, sản phẩm)
#   - df_customer: chỉ dòng có CustomerID (dùng cho phân tích khách hàng)

df_all = df_clean.copy()   # Dùng cho Phase 3 (doanh thu, sản phẩm)

df_customer = df_clean.dropna(subset=['CustomerID']).copy()
df_customer['CustomerID'] = df_customer['CustomerID'].astype(int)

print(f"df_all      (toàn bộ):    {len(df_all):,} dòng")
print(f"df_customer (có ID):      {len(df_customer):,} dòng")
print(f"Số khách hàng unique:     {df_customer['CustomerID'].nunique():,} khách")

# --- Chuẩn hóa tên nước ---
# Xem các giá trị bất thường
print("Tổng số quốc gia: ", df_all['Country'].nunique())
print("Liệt kê tất cả quốc gia: ", df_all['Country'].unique())

# Xóa khoảng trắng thừa đầu/cuối, viết hoa chữ đầu mỗi từ
df_all['Country'] = df_all['Country'].str.strip().str.title()
df_customer['Country'] = df_customer['Country'].str.strip().str.title()

# Kiểm tra lại
print(f"Số quốc gia sau chuẩn hóa: {df_all['Country'].nunique()}")

# --- Chuẩn hóa cột Description ---
df_all['Description'] = df_all['Description'].str.strip().str.upper()
df_customer['Description'] = df_customer['Description'].str.strip().str.upper()

print("Ví dụ Description sau chuẩn hóa:")
print(df_all['Description'].head(5).tolist())

# --- Tạo cột Revenue ---
df_all['Revenue'] = df_all['Quantity'] * df_all['UnitPrice']
df_customer['Revenue'] = df_customer['Quantity'] * df_customer['UnitPrice']

# --- Tách thời gian ---
for df_temp in [df_all, df_customer]:
    df_temp['InvoiceDate'] = pd.to_datetime(df_temp['InvoiceDate'])
    df_temp['Year']        = df_temp['InvoiceDate'].dt.year
    df_temp['Month']       = df_temp['InvoiceDate'].dt.month
    df_temp['DayOfWeek']   = df_temp['InvoiceDate'].dt.day_name()  # Monday, Tuesday,...
    df_temp['Hour']        = df_temp['InvoiceDate'].dt.hour
    df_temp['YearMonth']   = df_temp['InvoiceDate'].dt.to_period('M')

print("✅ Đã tạo thêm các cột: Revenue, Year, Month, DayOfWeek, Hour, YearMonth")
print(df_all[['InvoiceDate','Revenue','Year','Month','DayOfWeek','Hour']].head(3))


import random
random.seed(42)  # seed=42 để kết quả tái lập được (reproducibility)
np.random.seed(42)

# Lưu bản sạch trước khi inject
df_clean_backup = df_all.copy()

df_dirty = df_all.copy()
n = len(df_dirty)

print(f"Bắt đầu inject noise vào {n:,} dòng...\n")

# --- Noise 1: Thêm 5% ô trống vào cột Description ---
# Mô phỏng nhân viên bỏ trống khi nhập liệu
idx_desc = df_dirty.sample(frac=0.05, random_state=42).index
df_dirty.loc[idx_desc, 'Description'] = np.nan
print(f"Noise 1 — Thêm {len(idx_desc):,} ô trống vào Description (5%)")

# --- Noise 2: Thêm typo vào Country (2% dòng) ---
# Mô phỏng lỗi nhập tay: "United Kingdom" → "Untied Kingdom"
typo_map = {
    'United Kingdom': 'Untied Kingdom',
    'Germany': 'Gremany',
    'France': 'Farnce'
}
idx_country = df_dirty.sample(frac=0.02, random_state=99).index
df_dirty.loc[idx_country, 'Country'] = df_dirty.loc[idx_country, 'Country'].replace(typo_map)
print(f"Noise 2 — Thêm typo Country vào {len(idx_country):,} dòng (2%)")

# --- Noise 3: Thêm 1% duplicate rows ---
n_dup = int(n * 0.01)
dup_rows = df_dirty.sample(n=n_dup, random_state=7)
df_dirty = pd.concat([df_dirty, dup_rows], ignore_index=True)
print(f"Noise 3 — Thêm {n_dup:,} dòng trùng lặp (1%)")

# --- Noise 4: Outlier UnitPrice × 50 cho 0.3% dòng ---
idx_outlier = df_dirty.sample(frac=0.003, random_state=55).index
df_dirty.loc[idx_outlier, 'UnitPrice'] = df_dirty.loc[idx_outlier, 'UnitPrice'] * 50
print(f"Noise 4 — Thổi phồng UnitPrice × 50 cho {len(idx_outlier):,} dòng (0.3%)")

print(f"\n✅ df_dirty bây giờ có {len(df_dirty):,} dòng (sau khi thêm duplicate)")

def clean_pipeline(df_input):
    """
    Hàm này nhận vào dataframe bẩn → trả về dataframe sạch
    Dùng hàm để dễ tái sử dụng và trình bày trong paper
    """
    df = df_input.copy()
    report = {}  # Ghi lại những gì đã làm để báo cáo

    # Bước A: Xóa duplicate
    before = len(df)
    df = df.drop_duplicates()
    report['duplicates_removed'] = before - len(df)

    # Bước B: Sửa typo Country
    typo_fix = {
        'Untied Kingdom': 'United Kingdom',
        'Gremany': 'Germany',
        'Farnce': 'France'
    }
    df['Country'] = df['Country'].replace(typo_fix)
    df['Country'] = df['Country'].str.strip().str.title()
    report['typos_fixed'] = 'Country column normalized'

    # Bước C: Xử lý Description trống → điền "UNKNOWN"
    before_na = df['Description'].isnull().sum()
    df['Description'] = df['Description'].fillna('UNKNOWN')
    report['description_filled'] = before_na

    # Bước D: Xử lý outlier UnitPrice bằng IQR
    # IQR = khoảng giữa 25% và 75% → outlier là giá trị quá xa khoảng này
    Q1 = df['UnitPrice'].quantile(0.25)
    Q3 = df['UnitPrice'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3 * IQR  # Ngưỡng trên: 3 lần IQR
    before_outlier = (df['UnitPrice'] > upper_bound).sum()
    df = df[df['UnitPrice'] <= upper_bound]
    report['outliers_removed'] = before_outlier

    return df, report

# Chạy pipeline
df_recovered, report = clean_pipeline(df_dirty)

print("=== KẾT QUẢ PIPELINE LÀM SẠCH ===")
for key, val in report.items():
    print(f"  {key}: {val}")
print(f"\nDòng trước xử lý: {len(df_dirty):,}")
print(f"Dòng sau xử lý:   {len(df_recovered):,}")

# Lưu xuống file (tên khác file gốc → không đè online_retail.csv)
out_all = os.path.join('outputs', 'df_all_clean.csv')
out_cust = os.path.join('outputs', 'df_customer_clean.csv')
df_all.to_csv(out_all, index=False)
df_customer.to_csv(out_cust, index=False)

print("✅ Phase 2 hoàn thành!")
print(f"   {out_all}:      {len(df_all):,} dòng (tất cả giao dịch hợp lệ)")
print(f"   {out_cust}: {len(df_customer):,} dòng (chỉ khách có ID)")