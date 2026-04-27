"""
Export Advanced Analytics
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)

df_all = pd.read_csv('outputs/df_all_clean.csv', parse_dates=['InvoiceDate'])
df_customer = pd.read_csv('outputs/df_customer_clean.csv', parse_dates=['InvoiceDate'])
rfm = pd.read_csv('outputs/rfm_with_predictions.csv')

print("=== EXPORT ADVANCED ANALYTICS ===\n")

# ─── 1. Customer Lifetime Value (CLV) ───
reference_date = df_customer['InvoiceDate'].max() + pd.Timedelta(days=1)
customer_stats = df_customer.groupby('CustomerID').agg(
    first_purchase=('InvoiceDate', 'min'),
    last_purchase=('InvoiceDate', 'max'),
    total_orders=('InvoiceNo', 'nunique'),
    total_revenue=('Revenue', 'sum'),
    total_items=('Quantity', 'sum'),
    unique_products=('StockCode', 'nunique'),
    avg_order_value=('Revenue', lambda x: x.sum() / x.groupby(df_customer.loc[x.index, 'InvoiceNo']).ngroup() if len(x) > 0 else 0),
).reset_index()

customer_stats['customer_lifespan_days'] = (customer_stats['last_purchase'] - customer_stats['first_purchase']).dt.days
customer_stats['avg_days_between_orders'] = customer_stats['customer_lifespan_days'] / customer_stats['total_orders'].clip(lower=1)
customer_stats['avg_revenue_per_order'] = customer_stats['total_revenue'] / customer_stats['total_orders']
customer_stats['avg_items_per_order'] = customer_stats['total_items'] / customer_stats['total_orders']

avg_lifespan_months = customer_stats['customer_lifespan_days'].mean() / 30
avg_purchase_freq = customer_stats['total_orders'].mean()
avg_order_val = customer_stats['avg_revenue_per_order'].mean()
customer_stats['CLV_estimated'] = (customer_stats['avg_revenue_per_order'] * customer_stats['total_orders'] *
                                    (customer_stats['customer_lifespan_days'] / 365).clip(lower=1/12))

clv = customer_stats.merge(rfm[['CustomerID', 'Segment', 'ChurnProba']], on='CustomerID', how='left')
clv.to_csv('outputs/customer_lifetime_value.csv', index=False)
print(f"1. CLV: {len(clv):,} khách hàng -> customer_lifetime_value.csv")

# ─── 2. Cohort Analysis ───
df_customer['OrderMonth'] = df_customer['InvoiceDate'].dt.to_period('M')
first_month = df_customer.groupby('CustomerID')['OrderMonth'].min().reset_index()
first_month.columns = ['CustomerID', 'CohortMonth']
df_cohort = df_customer.merge(first_month, on='CustomerID')
df_cohort['CohortIndex'] = (df_cohort['OrderMonth'] - df_cohort['CohortMonth']).apply(lambda x: x.n)

cohort_data = df_cohort.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
cohort_data.columns = ['CohortMonth', 'CohortIndex', 'CustomerCount']

cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerCount')
cohort_size = cohort_pivot[0]
cohort_retention = cohort_pivot.divide(cohort_size, axis=0) * 100

cohort_retention_flat = cohort_retention.reset_index()
cohort_retention_flat['CohortMonth'] = cohort_retention_flat['CohortMonth'].astype(str)
cohort_retention_flat.to_csv('outputs/cohort_retention.csv', index=False)
print(f"2. Cohort Retention: {len(cohort_retention_flat)} cohorts -> cohort_retention.csv")

# ─── 3. Product ABC Classification (Pareto/80-15-5) ───
product_rev = df_all.groupby('Description').agg(
    total_revenue=('Revenue', 'sum'),
    total_quantity=('Quantity', 'sum'),
    total_orders=('InvoiceNo', 'nunique'),
    avg_unit_price=('UnitPrice', 'mean'),
).reset_index().sort_values('total_revenue', ascending=False)

product_rev['revenue_pct'] = product_rev['total_revenue'] / product_rev['total_revenue'].sum() * 100
product_rev['cumulative_pct'] = product_rev['revenue_pct'].cumsum()

def classify_abc(cum_pct):
    if cum_pct <= 80:
        return 'A'
    elif cum_pct <= 95:
        return 'B'
    return 'C'

product_rev['ABC_Class'] = product_rev['cumulative_pct'].apply(classify_abc)
product_rev['rank'] = range(1, len(product_rev) + 1)
product_rev.to_csv('outputs/product_abc_classification.csv', index=False)

a_count = (product_rev['ABC_Class'] == 'A').sum()
b_count = (product_rev['ABC_Class'] == 'B').sum()
c_count = (product_rev['ABC_Class'] == 'C').sum()
print(f"3. Product ABC: A={a_count}, B={b_count}, C={c_count} sản phẩm -> product_abc_classification.csv")

# ─── 4. Basket Size Analysis ───
basket = df_all.groupby('InvoiceNo').agg(
    items_count=('Quantity', 'sum'),
    unique_products=('StockCode', 'nunique'),
    basket_value=('Revenue', 'sum'),
    country=('Country', 'first'),
    date=('InvoiceDate', 'first'),
).reset_index()

basket['date'] = pd.to_datetime(basket['date'])
basket['month'] = basket['date'].dt.to_period('M').astype(str)

basket_monthly = basket.groupby('month').agg(
    avg_basket_value=('basket_value', 'mean'),
    avg_items=('items_count', 'mean'),
    avg_unique_products=('unique_products', 'mean'),
    total_orders=('InvoiceNo', 'count'),
    median_basket_value=('basket_value', 'median'),
).reset_index()
basket_monthly.to_csv('outputs/basket_analysis.csv', index=False)
print(f"4. Basket Analysis: {len(basket_monthly)} tháng -> basket_analysis.csv")

# ─── 5. Revenue Concentration (Pareto) ───
customer_rev = df_customer.groupby('CustomerID')['Revenue'].sum().sort_values(ascending=False).reset_index()
customer_rev['revenue_pct'] = customer_rev['Revenue'] / customer_rev['Revenue'].sum() * 100
customer_rev['cumulative_pct'] = customer_rev['revenue_pct'].cumsum()
customer_rev['customer_pct'] = np.arange(1, len(customer_rev) + 1) / len(customer_rev) * 100

top_20_pct_customers = len(customer_rev) * 0.2
revenue_top_20 = customer_rev.iloc[:int(top_20_pct_customers)]['revenue_pct'].sum()
customer_rev.to_csv('outputs/revenue_concentration.csv', index=False)
print(f"5. Revenue Concentration: Top 20% khách = {revenue_top_20:.1f}% doanh thu -> revenue_concentration.csv")

# ─── 6. Monthly Growth Rate ───
monthly = df_all.groupby(df_all['InvoiceDate'].dt.to_period('M').astype(str)).agg(
    revenue=('Revenue', 'sum'),
    orders=('InvoiceNo', 'nunique'),
    customers=('CustomerID', 'nunique') if 'CustomerID' in df_all.columns else ('InvoiceNo', 'nunique'),
    avg_order_value=('Revenue', 'mean'),
).reset_index()
monthly.columns = ['month', 'revenue', 'orders', 'customers', 'avg_order_value']
monthly['revenue_growth_pct'] = monthly['revenue'].pct_change() * 100
monthly['orders_growth_pct'] = monthly['orders'].pct_change() * 100
monthly.to_csv('outputs/monthly_growth.csv', index=False)
print(f"6. Monthly Growth: {len(monthly)} tháng -> monthly_growth.csv")

# ─── 7. Revenue by Segment ───
segment_rev = rfm.groupby('Segment').agg(
    customer_count=('CustomerID', 'count'),
    avg_monetary=('Monetary', 'mean'),
    total_monetary=('Monetary', 'sum'),
    avg_frequency=('Frequency', 'mean'),
    avg_recency=('Recency', 'mean'),
    avg_churn_proba=('ChurnProba', 'mean'),
).reset_index()
segment_rev['revenue_share_pct'] = segment_rev['total_monetary'] / segment_rev['total_monetary'].sum() * 100
segment_rev.to_csv('outputs/segment_revenue.csv', index=False)
print(f"7. Segment Revenue: {len(segment_rev)} nhóm -> segment_revenue.csv")

# ─── 8. Hourly-DayOfWeek Heatmap Data ───
heatmap = df_all.groupby(['DayOfWeek', 'Hour'])['Revenue'].sum().reset_index()
heatmap_pivot = heatmap.pivot(index='DayOfWeek', columns='Hour', values='Revenue')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
heatmap_pivot.to_csv('outputs/revenue_heatmap.csv')
print(f"8. Revenue Heatmap: DayOfWeek × Hour -> revenue_heatmap.csv")

# ─── 9. Country Performance ───
country_perf = df_all.groupby('Country').agg(
    total_revenue=('Revenue', 'sum'),
    total_orders=('InvoiceNo', 'nunique'),
    total_quantity=('Quantity', 'sum'),
    unique_products=('StockCode', 'nunique'),
    avg_order_value=('Revenue', lambda x: x.sum() / df_all.loc[x.index, 'InvoiceNo'].nunique()),
).reset_index().sort_values('total_revenue', ascending=False)
country_perf['revenue_share_pct'] = country_perf['total_revenue'] / country_perf['total_revenue'].sum() * 100
country_perf.to_csv('outputs/country_performance.csv', index=False)
print(f"9. Country Performance: {len(country_perf)} quốc gia -> country_performance.csv")

print("\n✅ Advanced Analytics export hoàn thành!")
print("   Tất cả file đã được lưu vào thư mục outputs/")
