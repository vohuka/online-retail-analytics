import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_all      = pd.read_csv('outputs/df_all_clean.csv', parse_dates=['InvoiceDate'])
df_customer = pd.read_csv('outputs/df_customer_clean.csv', parse_dates=['InvoiceDate'])

print(f"df_all:      {len(df_all):,} dòng")
print(f"df_customer: {len(df_customer):,} dòng")

# Doanh thu theo tháng
monthly = df_all.groupby('YearMonth')['Revenue'].sum().reset_index()
monthly['YearMonth'] = monthly['YearMonth'].astype(str)

plt.figure(figsize=(13, 5))
plt.plot(monthly['YearMonth'], monthly['Revenue'], marker='o',
         color='#2196F3', linewidth=2.5, markersize=7)
plt.fill_between(range(len(monthly)), monthly['Revenue'],
                 alpha=0.1, color='#2196F3')
plt.xticks(range(len(monthly)), monthly['YearMonth'], rotation=45)
plt.title('Doanh thu theo tháng', fontsize=14, fontweight='bold')
plt.ylabel('Doanh thu (£)')
for i, val in enumerate(monthly['Revenue']):
    plt.annotate(f'£{val:,.0f}', (i, val), textcoords='offset points',
                 xytext=(0, 6), ha='center', fontsize=7)
plt.tight_layout()
plt.savefig('outputs/monthly_revenue.png', dpi=150)

# In số liệu tháng đỉnh
peak = monthly.loc[monthly['Revenue'].idxmax()]
print(f"Tháng đỉnh: {peak['YearMonth']} với doanh thu £{peak['Revenue']:,.0f}")

# Doanh thu theo ngày trong tuần
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Sunday']
day_rev = df_all.groupby('DayOfWeek')['Revenue'].sum()
day_rev = day_rev.reindex(day_order)

plt.figure(figsize=(9, 4))
bars = plt.bar(day_rev.index, day_rev.values,
               color=['#FF6B6B' if v == day_rev.max() else '#90CAF9' for v in day_rev.values])
plt.title('Doanh thu theo ngày trong tuần', fontsize=13)
plt.ylabel('Doanh thu (£)')
for bar in bars:
    h = bar.get_height()
    plt.annotate(f'£{h:,.0f}',
                 (bar.get_x() + bar.get_width() / 2, h),
                 textcoords='offset points', xytext=(0, 4),
                 ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/day_revenue.png', dpi=150)

# Insight: Thứ 5 thường là ngày mạnh nhất

# Phân bổ theo giờ trong ngày
hour_rev = df_all.groupby('Hour')['Revenue'].sum()

plt.figure(figsize=(11, 4))
bars = plt.bar(hour_rev.index, hour_rev.values, color='#64B5F6')
plt.title('Doanh thu theo giờ trong ngày', fontsize=13)
plt.xlabel('Giờ (0–23)')
plt.ylabel('Doanh thu (£)')
plt.xticks(range(0, 24))
for bar in bars:
    h = bar.get_height()
    plt.annotate(f'£{h:,.0f}',
                 (bar.get_x() + bar.get_width() / 2, h),
                 textcoords='offset points', xytext=(0, 3),
                 ha='center', fontsize=6)
plt.tight_layout()
plt.savefig('outputs/hour_revenue.png', dpi=150)

# Insight: Đỉnh lúc 12:00 trưa

#! RFM analysis
# Ngày tham chiếu = ngày sau ngày cuối cùng trong dataset
reference_date = df_customer['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Ngày tham chiếu: {reference_date.date()}")

# Tính RFM cho từng khách hàng
rfm = df_customer.groupby('CustomerID').agg(
    Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency = ('InvoiceNo',   'nunique'),   # số hóa đơn unique
    Monetary  = ('Revenue',     'sum')
).reset_index()

# Thêm cột D = Product Diversity (số sản phẩm khác nhau đã mua)
diversity = df_customer.groupby('CustomerID')['StockCode'].nunique().reset_index()
diversity.columns = ['CustomerID', 'Diversity']
rfm = rfm.merge(diversity, on='CustomerID')

print("=== BẢNG RFM (5 dòng đầu) ===")
print(rfm.head())
print(f"\nTổng {len(rfm):,} khách hàng")
print(f"\nThống kê:")
print(rfm[['Recency','Frequency','Monetary','Diversity']].describe().round(1))

# Giải thích kết quả:
# Recency trung bình ~92 ngày = khách hàng trung bình mua lần cuối 92 ngày trước
# Frequency trung bình ~4.3 lần = mỗi khách mua trung bình 4.3 hóa đơn
# Monetary trung bình ~£1,887 = mỗi khách chi trung bình £1,887 trong 1 năm

# Vẽ phân phối 3 chỉ số
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col, color in zip(axes,
                           ['Recency','Frequency','Monetary'],
                           ['#EF5350','#42A5F5','#66BB6A']):
    # Dùng log scale vì phân phối lệch (skewed)
    data = np.log1p(rfm[col])  # log1p = log(x+1), tránh log(0)
    ax.hist(data, bins=40, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(f'Phân phối {col} (log scale)', fontweight='bold')
    ax.set_xlabel(f'log({col})')
    ax.set_ylabel('Số khách hàng')

plt.tight_layout()
plt.savefig('outputs/rfm_distribution.png', dpi=150)

# Bước 1: Log-transform + Chuẩn hóa (Best Practice cho RFM)
# Lý do áp dụng log-transform:
# - Monetary/Frequency có outlier cực mạnh (£280k vs £2k, 209 vs 2)
# - Log "nén" khoảng cách: log(280k) và log(2k) gần nhau hơn
# - Biến phân phối lệch (skewed) thành phân phối chuẩn hơn
# - K-Means hoạt động tốt hơn với dữ liệu chuẩn

features = ['Recency', 'Frequency', 'Monetary', 'Diversity']

# Bước 1a: Log-transform (dùng log1p = log(x+1) để tránh log(0))
rfm_log = rfm[features].copy()
for col in features:
    rfm_log[col] = np.log1p(rfm[col])

# Bước 1b: StandardScaler trên dữ liệu đã log
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

print("=== CHUYỂN ĐỔI DỮ LIỆU (5 khách đầu tiên) ===")
print("\n1. Dữ liệu gốc (raw):")
print(rfm[features].head().to_string())
print("\n2. Sau log-transform:")
print(rfm_log.head().round(3).to_string())
print("\n3. Sau StandardScaler (cuối cùng):")
print(pd.DataFrame(rfm_scaled[:5], columns=features).round(3).to_string())
print("\n→ Log giúp 'nén' outlier, Scale giúp các cột đóng góp đều vào K-Means")


# Bước 2: Dùng Elbow Method để tìm số nhóm tối ưu (k)
# "Elbow" = khuỷu tay: điểm mà thêm nhóm nữa không cải thiện đáng kể

inertia_list = []   # inertia = độ "lộn xộn" bên trong mỗi nhóm
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertia_list.append(km.inertia_)

plt.figure(figsize=(9, 4))
plt.plot(k_range, inertia_list, marker='o', color='#1565C0', linewidth=2)
plt.xlabel('Số nhóm k')
plt.ylabel('Inertia (càng thấp càng tốt)')
plt.title('Elbow Method — Tìm k tối ưu', fontsize=13)
plt.xticks(k_range)
plt.grid(alpha=0.3)
for k, inertia in zip(k_range, inertia_list):
    plt.annotate(f'{inertia:,.0f}', (k, inertia), textcoords='offset points',
                 xytext=(0, 6), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/elbow.png', dpi=150)

print("👆 Nhìn vào chỗ đường gấp 'khuỷu tay' — đó là k tối ưu (thường k=4 hoặc 5)")


# Bước 3: Kiểm tra thêm bằng Silhouette Score
# Silhouette = điểm đánh giá chất lượng phân nhóm (0→1, càng cao càng tốt)

sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    score = silhouette_score(rfm_scaled, labels)
    sil_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.4f}")

best_k = range(2, 8)[np.argmax(sil_scores)]
print(f"\n✅ k tốt nhất theo Silhouette: k = {best_k}")
print("💡 Sau log-transform, best_k thường cao hơn vì outlier đã được 'nén' lại")

# Bước 4: Chạy K-Means với k tốt nhất
# k = best_k
# Lưu ý: Nếu muốn chọn k thủ công (ví dụ k=4), uncomment dòng dưới:
k = 4

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Xem đặc điểm từng nhóm
cluster_summary = rfm.groupby('Cluster')[features].mean().round(1)
print("=== ĐẶC ĐIỂM TỪNG NHÓM ===")
print(cluster_summary)

# Bước 5: Đặt tên các nhóm dựa trên đặc điểm
# (Nhìn vào cluster_summary ở trên để phân tích)

def label_cluster(row):
    """
    Đặt tên nhóm dựa trên R, F, M:
    - Champions: R thấp (mua gần đây), F cao, M cao → khách VIP
    - Loyal: F cao, M trung bình → khách trung thành
    - At Risk: R cao (lâu không mua), từng M cao → sắp rời bỏ
    - Lost: R rất cao, F thấp, M thấp → đã rời bỏ
    """
    r_median = rfm['Recency'].median()
    f_median = rfm['Frequency'].median()
    m_median = rfm['Monetary'].median()

    if row['Recency'] <= r_median and row['Frequency'] >= f_median and row['Monetary'] >= m_median:
        return 'Champions'
    elif row['Frequency'] >= f_median and row['Monetary'] >= m_median:
        return 'Loyal Customers'
    elif row['Recency'] <= r_median:
        return 'Potential Loyalists'
    elif row['Recency'] > r_median * 1.5:
        return 'Lost / At Risk'
    else:
        return 'Need Attention'

rfm['Segment'] = rfm.apply(label_cluster, axis=1)

print("=== PHÂN PHỐI NHÓM KHÁCH HÀNG ===")
segment_counts = rfm['Segment'].value_counts()
print(segment_counts)
print(f"\nTổng: {segment_counts.sum():,} khách hàng")

# Vẽ biểu đồ phân bố nhóm
plt.figure(figsize=(9, 5))
colors = ['#4CAF50','#2196F3','#FF9800','#F44336','#9C27B0']
ax = segment_counts.plot(kind='bar', color=colors[:len(segment_counts)])
plt.title('Phân bố nhóm khách hàng (RFMD Clustering)', fontsize=13)
plt.xlabel('Nhóm')
plt.ylabel('Số khách hàng')
plt.xticks(rotation=30)
for container in ax.containers:
    ax.bar_label(container, fmt='{:,.0f}', padding=2, fontsize=9)
plt.tight_layout()
plt.savefig('outputs/customer_segments.png', dpi=150)


#! Market Basket Analysis
# Bước 1: Tạo "giỏ hàng" — mỗi hóa đơn là 1 giỏ
# Lọc UK vì chiếm 90%, tránh nhiễu từ nước khác
df_uk = df_all[df_all['Country'] == 'United Kingdom'].copy()

# Gom sản phẩm theo hóa đơn
basket = df_uk.groupby('InvoiceNo')['Description'].apply(list).reset_index()
transactions = basket['Description'].tolist()

print(f"Số hóa đơn (giỏ hàng): {len(transactions):,}")
print(f"Ví dụ giỏ hàng đầu tiên: {transactions[0][:5]}...")

# Bước 2: Chuyển sang dạng ma trận True/False
# Mỗi hàng = 1 hóa đơn, mỗi cột = 1 sản phẩm
# True = hóa đơn này có sản phẩm đó, False = không có

te = TransactionEncoder()
te_array = te.fit_transform(transactions)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

print(f"Ma trận: {basket_df.shape[0]:,} hóa đơn × {basket_df.shape[1]:,} sản phẩm")

# Bước 3: Chạy FP-Growth để tìm "tập sản phẩm phổ biến"
# min_support=0.02 nghĩa là: chỉ lấy combo xuất hiện trong ít nhất 2% hóa đơn

frequent_items = fpgrowth(basket_df,
                          min_support=0.02,
                          use_colnames=True)
frequent_items = frequent_items.sort_values('support', ascending=False)
print(f"Tìm được {len(frequent_items):,} tập sản phẩm phổ biến")
print(frequent_items.head(10))

# Bước 4: Tìm Association Rules (luật kết hợp)
# confidence=0.5 nghĩa là: nếu mua A thì có ít nhất 50% khả năng mua B

rules = association_rules(frequent_items,
                          metric='confidence',
                          min_threshold=0.5)
rules = rules.sort_values('lift', ascending=False)
# lift > 1 nghĩa là A và B thực sự liên quan (không phải ngẫu nhiên)

print(f"Tìm được {len(rules):,} luật kết hợp")
print("\nTop 10 luật mạnh nhất (lift cao nhất):")
print(rules[['antecedents','consequents','support','confidence','lift']].head(10).to_string())

# Giải thích output:
# antecedents → consequents: "Nếu mua X thì hay mua Y"
# support: combo này xuất hiện trong X% hóa đơn
# confidence: trong số hóa đơn có X, có Y% cũng có Y
# lift: X và Y liên quan nhau mạnh hơn ngẫu nhiên bao nhiêu lần

# Lưu bảng RFM với nhóm
rfm.to_csv('outputs/rfm_segments.csv', index=False)

# Lưu association rules
rules_save = rules.copy()
rules_save['antecedents'] = rules_save['antecedents'].astype(str)
rules_save['consequents'] = rules_save['consequents'].astype(str)
rules_save.to_csv('outputs/association_rules.csv', index=False)


print("✅ Phase 3 hoàn thành!")
print(f"   Đã phân {len(rfm):,} khách hàng thành {rfm['Segment'].nunique()} nhóm")
print(f"   Tìm được {len(rules):,} luật kết hợp sản phẩm")