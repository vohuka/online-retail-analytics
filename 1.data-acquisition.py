import os

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

os.makedirs('outputs', exist_ok=True)

# Cài đặt hiển thị cho đẹp
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Đọc file Excel vào biến df (viết tắt của DataFrame = bảng dữ liệu)
df = pd.read_csv('online_retail.csv')

# Xem kích thước: (số hàng, số cột)
print(f"Dataset có {df.shape[0]:,} hàng và {df.shape[1]} cột")
print(f"Tương đương {df.shape[0]:,} giao dịch!")

# Kiểm tra số ô trống mỗi cột
print("=== SỐ Ô TRỐNG MỖI CỘT ===")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Số ô trống': missing,
    'Tỉ lệ (%)': missing_pct
})
print(missing_df[missing_df['Số ô trống'] > 0])

# Hóa đơn hủy có InvoiceNo bắt đầu bằng 'C'
cancelled = df[df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"Số hóa đơn hủy: {len(cancelled):,}")
print(f"Tỉ lệ hủy: {len(cancelled)/len(df)*100:.2f}%")

negative_qty = df[df['Quantity'] < 0]
print(f"Số dòng có Quantity âm (Trả hàng lại): {len(negative_qty):,}")

# Xem thử vài dòng
negative_qty.head(3)

# .describe() = thống kê min, max, trung bình, ...
print(df[['Quantity', 'UnitPrice']].describe())

print("=== TOP 10 QUỐC GIA ===")
top_counts = df['Country'].value_counts().head(10)
top_countries_df = pd.DataFrame({
    'Số giao dịch': top_counts,
    'Tỉ lệ (%)': (top_counts / len(df) * 100).round(2),
})
print(top_countries_df)

# Vẽ biểu đồ top 10 quốc gia theo số giao dịch (có nhãn số trên mỗi cột)
fig, ax = plt.subplots(figsize=(10, 5))
top_counts.plot(kind='bar', color='steelblue', ax=ax)
ax.set_title('Top 10 quốc gia theo số giao dịch')
ax.set_xlabel('Quốc gia')
ax.set_ylabel('Số giao dịch')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
for container in ax.containers:
    ax.bar_label(container, fmt='{:,.0f}', padding=2, fontsize=8)
plt.tight_layout()
plt.savefig('outputs/top_countries.png', dpi=150)
plt.close()

# Vẽ bản đồ thế giới với tỉ lệ phần trăm
country_geo_df = df['Country'].value_counts().reset_index()
country_geo_df.columns = ['Country', 'Số giao dịch']
country_geo_df['Tỉ lệ (%)'] = (country_geo_df['Số giao dịch'] / len(df) * 100).round(2)

# Tạo thư mục 'outputs' nếu chưa có
os.makedirs('outputs', exist_ok=True)

fig = px.choropleth(country_geo_df,
                    locations="Country",
                    locationmode="country names",
                    color="Tỉ lệ (%)",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Tỉ lệ giao dịch theo quốc gia")
fig.write_html("outputs/country_transactions_map.html")

# Revenue = Quantity × UnitPrice (doanh thu mỗi dòng)
df['Revenue'] = df['Quantity'] * df['UnitPrice']
print(f"Tổng doanh thu thô: £{df['Revenue'].sum():,.2f}")

# Chuyển cột InvoiceDate sang kiểu datetime (ngày giờ)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Tách ra các cột phụ để phân tích
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')  # Ví dụ: 2011-03

print("Các tháng có trong dataset:")
print(sorted(df['YearMonth'].unique()))

# Tổng doanh thu mỗi tháng
monthly_revenue = df.groupby('YearMonth')['Revenue'].sum()

# Biểu đồ đường doanh thu theo tháng (nhãn giá trị tại mỗi đỉnh)
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(monthly_revenue))
ax.plot(x, monthly_revenue.values, marker='o', color='steelblue', linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels([str(p) for p in monthly_revenue.index], rotation=45, ha='right')
ax.set_title('Doanh thu theo tháng (Dec 2010 – Dec 2011)')
ax.set_xlabel('Tháng')
ax.set_ylabel('Doanh thu (£)')
for xi, yi in zip(x, monthly_revenue.values):
    ax.annotate(
        f'£{yi:,.0f}',
        (xi, yi),
        textcoords='offset points',
        xytext=(0, 6),
        ha='center',
        fontsize=7,
    )
plt.tight_layout()
plt.savefig('outputs/monthly_revenue.png', dpi=150)
plt.close()

# Tóm tắt phase 1 → ghi file (không in block này ra terminal)
top_country = df['Country'].value_counts().head(1)
top_country_name = top_country.index[0]
top_country_n = int(top_country.values[0])
top_country_pct = top_country_n / len(df) * 100

eda_lines = [
    '=' * 50,
    'TÓM TẮT PHASE 1 — KẾT QUẢ EDA',
    '=' * 50,
    f'Tổng số giao dịch:        {len(df):,}',
    f'Số khách có ID:           {df["CustomerID"].nunique():,}',
    f'Số sản phẩm unique:       {df["StockCode"].nunique():,}',
    f'Số quốc gia:              {df["Country"].nunique()}',
    f'Quốc gia chiếm đa số (chiếm tỉ lệ):     {top_country_name} ({top_country_n:,} giao dịch) - {top_country_pct:.2f}%',
    f'Khoảng thời gian:         {df["InvoiceDate"].min().date()} → {df["InvoiceDate"].max().date()}',
    f'Missing CustomerID:       {df["CustomerID"].isnull().sum():,} ({df["CustomerID"].isnull().mean()*100:.1f}%)',
    f"Hóa đơn hủy ('C'):        {df['InvoiceNo'].astype(str).str.startswith('C').sum():,}",
    f'Dòng Quantity âm:         {(df["Quantity"] < 0).sum():,}',
    f'Tổng doanh thu:       £{df["Revenue"].sum():,.0f}',
]
eda_path = os.path.join('outputs', 'EDA.txt')
with open(eda_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(eda_lines) + '\n')