# app.py — Premium BI Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS — MODERN DARK THEME =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d29;
    --bg-card: #1e2130;
    --bg-card-hover: #252940;
    --accent-blue: #4f8df9;
    --accent-cyan: #00d4aa;
    --accent-purple: #a78bfa;
    --accent-orange: #fb923c;
    --accent-pink: #f472b6;
    --accent-red: #ef4444;
    --accent-green: #22c55e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border-color: #2d3348;
    --gradient-blue: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-green: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --gradient-orange: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-cyan: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
}

[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    padding: 1.5rem 2rem 2rem;
    max-width: 1400px;
}

/* Header */
.dashboard-header {
    background: linear-gradient(135deg, #1e2130 0%, #2d1b69 50%, #1a1d29 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dashboard-header::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 300px; height: 100%;
    background: radial-gradient(circle at 80% 50%, rgba(79,141,249,0.12) 0%, transparent 70%);
}
.dashboard-header h1 {
    color: var(--text-primary);
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.dashboard-header p {
    color: var(--text-secondary);
    font-size: 14px;
    margin: 0;
}

/* KPI Cards */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 22px 24px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    border-color: var(--accent-blue);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(79,141,249,0.15);
}
.kpi-card .kpi-icon {
    font-size: 28px;
    margin-bottom: 10px;
    display: block;
}
.kpi-card .kpi-label {
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.kpi-card .kpi-value {
    color: var(--text-primary);
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.kpi-card .kpi-sub {
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 4px;
}

/* Section Headers */
.section-header {
    color: var(--text-primary);
    font-size: 18px;
    font-weight: 600;
    margin: 28px 0 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 14px;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 14px 18px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 20px;
}

/* Plotly chart containers */
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Info/Warning boxes */
.stAlert {
    border-radius: 12px;
}

/* Expanders */
.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--text-primary);
}

/* Dividers */
hr {
    border-color: var(--border-color);
}
</style>
""", unsafe_allow_html=True)

# ===== PLOTLY THEME =====
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(30,33,48,0.6)',
    font=dict(family='Inter, sans-serif', color='#e2e8f0', size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor='rgba(45,51,72,0.6)', zerolinecolor='rgba(45,51,72,0.6)'),
    yaxis=dict(gridcolor='rgba(45,51,72,0.6)', zerolinecolor='rgba(45,51,72,0.6)'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    hoverlabel=dict(bgcolor='#1e2130', font_size=12, font_family='Inter'),
)

COLORS = {
    'blue': '#4f8df9',
    'cyan': '#00d4aa',
    'purple': '#a78bfa',
    'orange': '#fb923c',
    'pink': '#f472b6',
    'red': '#ef4444',
    'green': '#22c55e',
    'yellow': '#fbbf24',
    'slate': '#94a3b8',
}

COLOR_SEQUENCE = ['#4f8df9', '#00d4aa', '#a78bfa', '#fb923c', '#f472b6',
                  '#22c55e', '#fbbf24', '#ef4444', '#06b6d4', '#8b5cf6']


# ===== LOAD DỮ LIỆU =====
@st.cache_data
def load_data():
    df_all = pd.read_csv('outputs/df_all_clean.csv', parse_dates=['InvoiceDate'])
    df_customer = pd.read_csv('outputs/df_customer_clean.csv', parse_dates=['InvoiceDate'])
    rfm = pd.read_csv('outputs/rfm_with_predictions.csv')
    return df_all, df_customer, rfm

@st.cache_data
def load_association_rules():
    try:
        rules = pd.read_csv('outputs/association_rules.csv')
        rules['antecedents'] = rules['antecedents'].str.replace("frozenset\\(\\{", "", regex=True).str.replace("\\}\\)", "", regex=True).str.replace("'", "")
        rules['consequents'] = rules['consequents'].str.replace("frozenset\\(\\{", "", regex=True).str.replace("\\}\\)", "", regex=True).str.replace("'", "")
        return rules
    except Exception:
        return None

@st.cache_data
def load_forecast():
    try:
        return pd.read_csv('outputs/forecast_results.csv', parse_dates=['ds'])
    except Exception:
        return None

@st.cache_data
def load_forecast_summary():
    try:
        return pd.read_csv('outputs/best_forecast_models_summary.csv')
    except Exception:
        return None

@st.cache_data
def load_advanced_analytics():
    data = {}
    files = {
        'clv': 'customer_lifetime_value.csv',
        'cohort': 'cohort_retention.csv',
        'product_abc': 'product_abc_classification.csv',
        'basket': 'basket_analysis.csv',
        'concentration': 'revenue_concentration.csv',
        'monthly_growth': 'monthly_growth.csv',
        'segment_revenue': 'segment_revenue.csv',
        'heatmap': 'revenue_heatmap.csv',
        'country_perf': 'country_performance.csv',
    }
    for key, filename in files.items():
        path = f'outputs/{filename}'
        try:
            if key == 'heatmap':
                data[key] = pd.read_csv(path, index_col=0)
            else:
                data[key] = pd.read_csv(path)
        except Exception:
            data[key] = None
    return data


df_all, df_customer, rfm = load_data()
rules_df = load_association_rules()
forecast_df = load_forecast()
forecast_summary = load_forecast_summary()
advanced = load_advanced_analytics()

# ===== HEADER =====
st.markdown("""
<div class="dashboard-header">
    <h1>📊 Online Retail — Analytics Dashboard</h1>
    <p>UCI Online Retail Dataset · Dec 2010 – Dec 2011 · UK-based giftware retailer · Powered by AI & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ===== TABS =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tổng quan",
    "👥 Khách hàng & Association Rules",
    "📈 Dự báo & Mô hình",
    "🔬 Phân tích nâng cao",
    "🤖 Chatbot BI"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1: TỔNG QUAN
# ══════════════════════════════════════════════════════════════════
with tab1:
    total_revenue = df_all['Revenue'].sum()
    total_orders = df_all['InvoiceNo'].nunique()
    total_customers = df_customer['CustomerID'].nunique()
    total_products = df_all['StockCode'].nunique()
    avg_order_value = total_revenue / total_orders

    # KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <span class="kpi-icon">💰</span>
            <div class="kpi-label">Tổng doanh thu</div>
            <div class="kpi-value">£{total_revenue:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <span class="kpi-icon">📦</span>
            <div class="kpi-label">Số hóa đơn</div>
            <div class="kpi-value">{total_orders:,}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <span class="kpi-icon">👤</span>
            <div class="kpi-label">Khách hàng</div>
            <div class="kpi-value">{total_customers:,}</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <span class="kpi-icon">🏷️</span>
            <div class="kpi-label">Sản phẩm</div>
            <div class="kpi-value">{total_products:,}</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card">
            <span class="kpi-icon">🧾</span>
            <div class="kpi-label">Giá trị đơn TB</div>
            <div class="kpi-value">£{avg_order_value:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue trend
    monthly = df_all.groupby(df_all['InvoiceDate'].dt.to_period('M').astype(str))['Revenue'].sum().reset_index()
    monthly.columns = ['Tháng', 'Doanh thu']

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=monthly['Tháng'], y=monthly['Doanh thu'],
        mode='lines+markers',
        line=dict(color=COLORS['blue'], width=3),
        marker=dict(size=8, color=COLORS['blue'], line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(79,141,249,0.08)',
        name='Doanh thu',
        hovertemplate='%{x}<br>£%{y:,.0f}<extra></extra>'
    ))
    fig_monthly.update_layout(**PLOTLY_LAYOUT, title='Doanh thu theo tháng', height=380,
                               xaxis_tickangle=-45)
    st.plotly_chart(fig_monthly, width='stretch')

    # Top products + Top countries
    col_left, col_right = st.columns(2)

    with col_left:
        top_products = df_all.groupby('Description')['Revenue'].sum().nlargest(10).reset_index()
        fig_prod = px.bar(top_products.sort_values('Revenue'),
                          x='Revenue', y='Description', orientation='h',
                          color_discrete_sequence=[COLORS['cyan']],
                          title='Top 10 sản phẩm theo doanh thu')
        fig_prod.update_layout(**PLOTLY_LAYOUT, height=420)
        fig_prod.update_traces(hovertemplate='%{y}<br>£%{x:,.0f}<extra></extra>')
        st.plotly_chart(fig_prod, width='stretch')

    with col_right:
        top_countries = df_all.groupby('Country')['Revenue'].sum().nlargest(10).reset_index()
        fig_country = px.bar(top_countries.sort_values('Revenue'),
                             x='Revenue', y='Country', orientation='h',
                             color_discrete_sequence=[COLORS['orange']],
                             title='Top 10 quốc gia theo doanh thu')
        fig_country.update_layout(**PLOTLY_LAYOUT, height=420)
        fig_country.update_traces(hovertemplate='%{y}<br>£%{x:,.0f}<extra></extra>')
        st.plotly_chart(fig_country, width='stretch')

    # Revenue heatmap
    if advanced.get('heatmap') is not None:
        st.markdown('<div class="section-header">🔥 Heatmap doanh thu theo Giờ × Ngày trong tuần</div>', unsafe_allow_html=True)
        heatmap_data = advanced['heatmap']
        fig_heat = px.imshow(
            heatmap_data.values,
            labels=dict(x="Giờ", y="Ngày", color="Doanh thu (£)"),
            x=[str(c) for c in heatmap_data.columns],
            y=list(heatmap_data.index),
            color_continuous_scale='Viridis',
            title='Doanh thu theo Giờ & Ngày trong tuần'
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_heat, width='stretch')


# ══════════════════════════════════════════════════════════════════
# TAB 2: KHÁCH HÀNG & ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Phân nhóm khách hàng — RFMD Clustering</div>', unsafe_allow_html=True)

    segment_counts = rfm['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Nhóm', 'Số khách']

    col_a, col_b = st.columns(2)
    with col_a:
        fig_donut = px.pie(segment_counts, values='Số khách', names='Nhóm',
                           hole=0.45, title='Phân bố nhóm khách hàng',
                           color_discrete_sequence=COLOR_SEQUENCE)
        fig_donut.update_layout(**PLOTLY_LAYOUT, height=420)
        fig_donut.update_traces(textposition='inside', textinfo='percent+label',
                                textfont_size=11)
        st.plotly_chart(fig_donut, width='stretch')

    with col_b:
        fig_scatter = px.scatter(rfm, x='Frequency', y='Monetary',
                                 color='Segment', size='Diversity',
                                 hover_data=['CustomerID'],
                                 title='Biểu đồ RFM — Frequency vs Monetary',
                                 opacity=0.65,
                                 color_discrete_sequence=COLOR_SEQUENCE)
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig_scatter, width='stretch')

    # Segment summary table
    if advanced.get('segment_revenue') is not None:
        st.markdown('<div class="section-header">💎 Thống kê theo nhóm khách hàng</div>', unsafe_allow_html=True)
        seg_rev = advanced['segment_revenue'].copy()
        seg_rev.columns = ['Nhóm', 'Số khách', 'Chi tiêu TB (£)', 'Tổng chi tiêu (£)',
                           'Tần suất TB', 'Recency TB (ngày)', 'Xác suất churn TB', 'Tỷ trọng DT (%)']
        for col in ['Chi tiêu TB (£)', 'Tổng chi tiêu (£)']:
            seg_rev[col] = seg_rev[col].apply(lambda x: f"£{x:,.0f}")
        seg_rev['Xác suất churn TB'] = seg_rev['Xác suất churn TB'].apply(lambda x: f"{x*100:.1f}%")
        seg_rev['Tỷ trọng DT (%)'] = seg_rev['Tỷ trọng DT (%)'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(seg_rev, width='stretch', hide_index=True)

    # Churn table
    st.markdown('<div class="section-header">⚠️ Top 20 khách hàng có nguy cơ churn cao nhất</div>', unsafe_allow_html=True)
    churn_risk = rfm.nlargest(20, 'ChurnProba')[
        ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment', 'ChurnProba']
    ].copy()
    churn_risk['ChurnProba'] = (churn_risk['ChurnProba'] * 100).round(1).astype(str) + '%'
    churn_risk['Monetary'] = churn_risk['Monetary'].apply(lambda x: f"£{x:,.0f}")
    churn_risk.columns = ['Mã KH', 'Recency (ngày)', 'Tần suất', 'Tổng chi tiêu', 'Nhóm', 'Xác suất churn']
    st.dataframe(churn_risk, width='stretch', hide_index=True)

    # ─── ASSOCIATION RULES ───
    st.markdown('<div class="section-header">🔗 Association Rules — Quy luật mua hàng kết hợp</div>', unsafe_allow_html=True)
    st.caption("Phân tích Market Basket: 'Nếu mua sản phẩm A → thường mua thêm sản phẩm B'. Dùng để xây dựng combo khuyến mãi & cross-sell.")

    if rules_df is not None and len(rules_df) > 0:
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("**🔝 Top 10 luật mạnh nhất** *(Lift cao = liên quan mạnh)*")
            top_rules = rules_df.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            top_rules.columns = ['Nếu mua...', 'Thường mua thêm...', 'Support', 'Confidence', 'Lift']
            top_rules['Support'] = (top_rules['Support'] * 100).round(2).astype(str) + '%'
            top_rules['Confidence'] = (top_rules['Confidence'] * 100).round(1).astype(str) + '%'
            top_rules['Lift'] = top_rules['Lift'].round(2)
            st.dataframe(top_rules, width='stretch', hide_index=True)

        with col_r2:
            st.markdown("**📉 Top 10 luật yếu nhất** *(Lift thấp nhất nhưng > 1)*")
            bottom_rules = rules_df[rules_df['lift'] > 1].nsmallest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
            bottom_rules.columns = ['Nếu mua...', 'Thường mua thêm...', 'Support', 'Confidence', 'Lift']
            bottom_rules['Support'] = (bottom_rules['Support'] * 100).round(2).astype(str) + '%'
            bottom_rules['Confidence'] = (bottom_rules['Confidence'] * 100).round(1).astype(str) + '%'
            bottom_rules['Lift'] = bottom_rules['Lift'].round(2)
            st.dataframe(bottom_rules, width='stretch', hide_index=True)

        with st.expander("📖 Cách đọc Association Rules & Gợi ý khuyến mãi"):
            st.markdown("""
**Giải thích các chỉ số:**
- **Support**: % hóa đơn có chứa combo này (càng cao = càng phổ biến)
- **Confidence**: Xác suất mua B khi đã mua A (càng cao = càng chắc chắn)
- **Lift**: Mức liên quan so với ngẫu nhiên (>1 = có mối quan hệ thực sự, càng cao càng mạnh)

**Gợi ý chiến lược khuyến mãi:**
1. **Bundle Deal**: Gom các sản phẩm có Lift cao thành combo giảm 10-15%
2. **Cross-sell**: Khi khách thêm sản phẩm A vào giỏ → gợi ý sản phẩm B với giá ưu đãi
3. **Upsell**: Combo có Confidence > 70% → đặt cạnh nhau trên kệ/web
4. **Email Marketing**: Gửi deal combo cho khách đã mua 1 trong 2 sản phẩm
            """)

        # Visualization: top rules as bar chart
        top_viz = rules_df.nlargest(15, 'lift').copy()
        top_viz['rule'] = top_viz['antecedents'].str[:30] + ' → ' + top_viz['consequents'].str[:30]
        fig_rules = px.bar(top_viz.sort_values('lift'),
                           x='lift', y='rule', orientation='h',
                           color='confidence',
                           color_continuous_scale='Viridis',
                           title='Top 15 Association Rules theo Lift',
                           labels={'lift': 'Lift Score', 'rule': 'Luật', 'confidence': 'Confidence'})
        fig_rules.update_layout(**PLOTLY_LAYOUT, height=500)
        st.plotly_chart(fig_rules, width='stretch')
    else:
        st.warning("Chưa có file association_rules.csv — chạy data-mining.py trước")


# ══════════════════════════════════════════════════════════════════
# TAB 3: DỰ BÁO & MÔ HÌNH
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📈 Sales Forecasting — Dự báo doanh thu</div>', unsafe_allow_html=True)

    # Model comparison summary
    if forecast_summary is not None:
        st.markdown("**So sánh mô hình dự báo tốt nhất (Prophet vs XGBoost)**")
        summary_display = forecast_summary.copy()
        summary_display.columns = ['Thuật toán', 'Phương pháp tốt nhất', 'MAPE (%)', 'File Model']
        st.dataframe(summary_display, width='stretch', hide_index=True)
        st.caption("MAPE = Mean Absolute Percentage Error — sai số phần trăm trung bình. Càng thấp càng chính xác.")

    st.divider()

    # Forecast charts
    if forecast_df is not None and len(forecast_df) > 0:
        model_choice = st.selectbox(
            "Chọn mô hình",
            forecast_df['model'].unique().tolist(),
            key='forecast_model'
        )
        approach_choice = st.selectbox(
            "Chọn phương pháp",
            forecast_df[forecast_df['model'] == model_choice]['approach'].unique().tolist(),
            key='forecast_approach'
        )

        subset = forecast_df[(forecast_df['model'] == model_choice) & (forecast_df['approach'] == approach_choice)]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=subset['ds'], y=subset['actual'],
            name='Thực tế', line=dict(color=COLORS['green'], width=2.5),
            mode='lines+markers', marker=dict(size=6),
            hovertemplate='%{x|%d/%m/%Y}<br>Thực tế: £%{y:,.0f}<extra></extra>'
        ))
        fig_fc.add_trace(go.Scatter(
            x=subset['ds'], y=subset['predicted'],
            name='Dự báo', line=dict(color=COLORS['red'], width=2.5, dash='dash'),
            mode='lines+markers', marker=dict(size=6),
            hovertemplate='%{x|%d/%m/%Y}<br>Dự báo: £%{y:,.0f}<extra></extra>'
        ))
        if 'predicted_lower' in subset.columns and 'predicted_upper' in subset.columns:
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([subset['ds'], subset['ds'][::-1]]),
                y=pd.concat([subset['predicted_upper'], subset['predicted_lower'][::-1]]),
                fill='toself', fillcolor='rgba(239,68,68,0.1)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Khoảng tin cậy', showlegend=True,
                hoverinfo='skip'
            ))

        mae_val = np.mean(np.abs(subset['actual'] - subset['predicted']))
        mape_val = np.mean(np.abs((subset['actual'] - subset['predicted']) / subset['actual'])) * 100

        fig_fc.update_layout(
            **PLOTLY_LAYOUT,
            title=f'Dự báo doanh thu — {model_choice} ({approach_choice}) | MAE: £{mae_val:,.0f} | MAPE: {mape_val:.1f}%',
            height=450
        )
        st.plotly_chart(fig_fc, width='stretch')
    else:
        st.info("Chưa có dữ liệu forecast. Hãy chạy lại `predictive-modeling.py` để export `forecast_results.csv`.")

        # Fallback: show saved images
        for img_path, title in [('outputs/forecast_prophet.png', 'Prophet — Cách 1'),
                                 ('outputs/forecast_prophet_v2.png', 'Prophet — Cách 2')]:
            if os.path.exists(img_path):
                st.image(img_path, caption=title, width='stretch')

    st.divider()

    # Churn Prediction section
    st.markdown('<div class="section-header">🎯 Churn Prediction — Dự báo khách rời bỏ</div>', unsafe_allow_html=True)

    churn_counts = rfm['Churned'].value_counts() if 'Churned' in rfm.columns else None
    if churn_counts is not None:
        c1, c2, c3 = st.columns(3)
        active = churn_counts.get(0, 0)
        churned = churn_counts.get(1, 0)
        churn_rate = churned / (active + churned) * 100
        c1.metric("Còn active", f"{active:,}")
        c2.metric("Đã churn", f"{churned:,}")
        c3.metric("Tỷ lệ churn", f"{churn_rate:.1f}%")

    col_cm, col_fi = st.columns(2)
    with col_cm:
        if os.path.exists('outputs/confusion_matrix.png'):
            st.image('outputs/confusion_matrix.png', caption='Confusion Matrix — Churn Prediction', width='stretch')
    with col_fi:
        if os.path.exists('outputs/feature_importance.png'):
            st.image('outputs/feature_importance.png', caption='Feature Importance', width='stretch')


# ══════════════════════════════════════════════════════════════════
# TAB 4: PHÂN TÍCH NÂNG CAO
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🔬 Phân tích chiến lược kinh doanh nâng cao</div>', unsafe_allow_html=True)
    st.caption("Dữ liệu phân tích chuyên sâu dành cho đội ngũ chiến lược — CLV, Cohort, Pareto, ABC Classification")

    # ─── Monthly Growth ───
    if advanced.get('monthly_growth') is not None:
        mg = advanced['monthly_growth']
        st.markdown('<div class="section-header">📈 Tăng trưởng doanh thu theo tháng (MoM)</div>', unsafe_allow_html=True)

        fig_growth = make_subplots(specs=[[{"secondary_y": True}]])
        fig_growth.add_trace(go.Bar(
            x=mg['month'], y=mg['revenue'],
            name='Doanh thu (£)', marker_color=COLORS['blue'], opacity=0.7,
            hovertemplate='%{x}<br>£%{y:,.0f}<extra></extra>'
        ), secondary_y=False)
        fig_growth.add_trace(go.Scatter(
            x=mg['month'], y=mg['revenue_growth_pct'],
            name='Tăng trưởng MoM (%)', line=dict(color=COLORS['orange'], width=2.5),
            mode='lines+markers', marker=dict(size=7),
            hovertemplate='%{x}<br>%{y:+.1f}%<extra></extra>'
        ), secondary_y=True)
        fig_growth.update_layout(**PLOTLY_LAYOUT, title='Doanh thu & Tốc độ tăng trưởng MoM',
                                  height=400, xaxis_tickangle=-45)
        fig_growth.update_yaxes(title_text="Doanh thu (£)", secondary_y=False)
        fig_growth.update_yaxes(title_text="Tăng trưởng (%)", secondary_y=True)
        st.plotly_chart(fig_growth, width='stretch')

    # ─── Revenue Concentration (Pareto) ───
    if advanced.get('concentration') is not None:
        conc = advanced['concentration']
        st.markdown('<div class="section-header">📊 Phân tích Pareto — Tập trung doanh thu</div>', unsafe_allow_html=True)

        top20_rev = conc[conc['customer_pct'] <= 20]['revenue_pct'].sum()

        p1, p2, p3 = st.columns(3)
        p1.metric("Top 20% khách hàng đóng góp", f"{top20_rev:.1f}% doanh thu")
        p2.metric("Top 10% khách hàng đóng góp", f"{conc[conc['customer_pct'] <= 10]['revenue_pct'].sum():.1f}% doanh thu")
        p3.metric("Top 1% khách hàng đóng góp", f"{conc[conc['customer_pct'] <= 1]['revenue_pct'].sum():.1f}% doanh thu")

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Scatter(
            x=conc['customer_pct'], y=conc['cumulative_pct'],
            mode='lines', line=dict(color=COLORS['blue'], width=2.5),
            fill='tozeroy', fillcolor='rgba(79,141,249,0.1)',
            name='Doanh thu tích lũy',
            hovertemplate='Top %{x:.0f}% khách hàng<br>= %{y:.1f}% doanh thu<extra></extra>'
        ))
        fig_pareto.add_hline(y=80, line_dash="dash", line_color=COLORS['red'],
                              annotation_text="80% doanh thu")
        fig_pareto.update_layout(**PLOTLY_LAYOUT, title='Đường cong Pareto — Tập trung doanh thu',
                                  height=400, xaxis_title='% Khách hàng (xếp theo doanh thu giảm dần)',
                                  yaxis_title='% Doanh thu tích lũy')
        st.plotly_chart(fig_pareto, width='stretch')

    # ─── Product ABC ───
    if advanced.get('product_abc') is not None:
        abc = advanced['product_abc']
        st.markdown('<div class="section-header">🏷️ Phân loại sản phẩm ABC</div>', unsafe_allow_html=True)

        abc_summary = abc.groupby('ABC_Class').agg(
            count=('Description', 'count'),
            total_rev=('total_revenue', 'sum')
        ).reset_index()
        abc_summary['rev_pct'] = (abc_summary['total_rev'] / abc_summary['total_rev'].sum() * 100).round(1)

        a1, a2, a3 = st.columns(3)
        for col, cls, color, icon in [(a1, 'A', COLORS['green'], '🟢'),
                                       (a2, 'B', COLORS['orange'], '🟡'),
                                       (a3, 'C', COLORS['red'], '🔴')]:
            row = abc_summary[abc_summary['ABC_Class'] == cls]
            if len(row) > 0:
                col.metric(f"{icon} Nhóm {cls}", f"{row['count'].values[0]:,} sản phẩm",
                           f"{row['rev_pct'].values[0]:.1f}% doanh thu")

        st.caption("A = Top 80% doanh thu (VIP products) | B = 80-95% (cần duy trì) | C = 95-100% (xem xét loại bỏ)")

        top_a = abc[abc['ABC_Class'] == 'A'].head(15)
        fig_abc = px.bar(top_a.sort_values('total_revenue'),
                         x='total_revenue', y='Description', orientation='h',
                         color='ABC_Class',
                         color_discrete_map={'A': COLORS['green'], 'B': COLORS['orange'], 'C': COLORS['red']},
                         title='Top 15 sản phẩm nhóm A (VIP)')
        fig_abc.update_layout(**PLOTLY_LAYOUT, height=480)
        fig_abc.update_traces(hovertemplate='%{y}<br>£%{x:,.0f}<extra></extra>')
        st.plotly_chart(fig_abc, width='stretch')

    # ─── Basket Analysis ───
    if advanced.get('basket') is not None:
        bk = advanced['basket']
        st.markdown('<div class="section-header">🛒 Phân tích giỏ hàng</div>', unsafe_allow_html=True)

        fig_basket = make_subplots(specs=[[{"secondary_y": True}]])
        fig_basket.add_trace(go.Bar(
            x=bk['month'], y=bk['avg_basket_value'],
            name='Giá trị giỏ hàng TB (£)', marker_color=COLORS['purple'], opacity=0.7
        ), secondary_y=False)
        fig_basket.add_trace(go.Scatter(
            x=bk['month'], y=bk['total_orders'],
            name='Số đơn hàng', line=dict(color=COLORS['cyan'], width=2),
            mode='lines+markers'
        ), secondary_y=True)
        fig_basket.update_layout(**PLOTLY_LAYOUT, title='Giá trị giỏ hàng & Số đơn theo tháng',
                                  height=380, xaxis_tickangle=-45)
        st.plotly_chart(fig_basket, width='stretch')

    # ─── Cohort Retention ───
    if advanced.get('cohort') is not None:
        cohort = advanced['cohort']
        st.markdown('<div class="section-header">📅 Cohort Retention — Tỷ lệ giữ chân khách hàng</div>', unsafe_allow_html=True)
        st.caption("Mỗi dòng = nhóm khách hàng mua lần đầu cùng tháng. Cột = tháng thứ N kể từ lần mua đầu. Giá trị = % khách quay lại.")

        cohort_display = cohort.set_index('CohortMonth')
        numeric_cols = [c for c in cohort_display.columns if c not in ['CohortMonth']]
        cohort_vals = cohort_display[numeric_cols].values

        fig_cohort = px.imshow(
            cohort_vals,
            labels=dict(x="Tháng thứ N", y="Cohort (tháng mua đầu tiên)", color="Retention (%)"),
            x=[f"T{i}" for i in range(cohort_vals.shape[1])],
            y=list(cohort_display.index),
            color_continuous_scale='RdYlGn',
            title='Cohort Retention Heatmap'
        )
        fig_cohort.update_layout(**PLOTLY_LAYOUT, height=450)
        st.plotly_chart(fig_cohort, width='stretch')

    # ─── Country Performance ───
    if advanced.get('country_perf') is not None:
        cp = advanced['country_perf']
        st.markdown('<div class="section-header">🌍 Hiệu suất theo quốc gia</div>', unsafe_allow_html=True)

        fig_map = px.choropleth(
            cp, locations='Country', locationmode='country names',
            color='total_revenue', hover_name='Country',
            hover_data={'total_orders': ':,', 'revenue_share_pct': ':.1f'},
            color_continuous_scale='Viridis',
            title='Doanh thu theo quốc gia'
        )
        fig_map.update_layout(**PLOTLY_LAYOUT, height=450, geo=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_map, width='stretch')


# ══════════════════════════════════════════════════════════════════
# TAB 5: CHATBOT BI
# ══════════════════════════════════════════════════════════════════

def get_ai_response(question: str, df_all: pd.DataFrame, rfm: pd.DataFrame) -> str:
    try:
        import anthropic
        summary_stats = build_data_summary(df_all, rfm)
        context = f"""Bạn là một Senior Business Intelligence Analyst. Bạn có dữ liệu tổng hợp sau:

{summary_stats}

Dữ liệu chi tiết:
- df_all: {len(df_all):,} giao dịch (InvoiceNo, StockCode, Description, Quantity, UnitPrice, Revenue, InvoiceDate, Country, Month, DayOfWeek, Hour)
- rfm: {len(rfm):,} khách hàng (CustomerID, Recency, Frequency, Monetary, Diversity, Segment, ChurnProba)

Câu hỏi: {question}

Trả lời bằng tiếng Việt, chính xác dựa trên dữ liệu. Không bịa số liệu."""

        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": context}]
        )
        return message.content[0].text
    except Exception:
        return smart_query(question, df_all, rfm)


def build_data_summary(df_all, rfm):
    total_revenue = df_all['Revenue'].sum()
    total_orders = df_all['InvoiceNo'].nunique()
    total_customers = rfm['CustomerID'].nunique()
    total_products = df_all['StockCode'].nunique()
    avg_order_value = total_revenue / total_orders

    top5_products = df_all.groupby('Description')['Revenue'].sum().nlargest(5)
    top5_countries = df_all.groupby('Country')['Revenue'].sum().nlargest(5)
    segment_dist = rfm['Segment'].value_counts()
    churn_high = rfm[rfm['ChurnProba'] > 0.7]

    summary = f"""
TỔNG QUAN:
- Tổng doanh thu: £{total_revenue:,.0f}
- Số hóa đơn: {total_orders:,}
- Số khách hàng (có ID): {total_customers:,}
- Số sản phẩm: {total_products:,}
- Giá trị đơn hàng trung bình: £{avg_order_value:,.0f}
- Khoảng thời gian: {df_all['InvoiceDate'].min().date()} đến {df_all['InvoiceDate'].max().date()}

TOP 5 SẢN PHẨM THEO DOANH THU:
"""
    for i, (name, rev) in enumerate(top5_products.items(), 1):
        summary += f"  {i}. {name}: £{rev:,.0f}\n"

    summary += "\nTOP 5 QUỐC GIA THEO DOANH THU:\n"
    for i, (name, rev) in enumerate(top5_countries.items(), 1):
        summary += f"  {i}. {name}: £{rev:,.0f}\n"

    summary += "\nPHÂN NHÓM KHÁCH HÀNG:\n"
    for seg, count in segment_dist.items():
        summary += f"  - {seg}: {count:,} khách\n"

    summary += f"\nKHÁCH CÓ NGUY CƠ CHURN CAO (>70%): {len(churn_high):,} khách"

    return summary


def smart_query(question: str, df_all, rfm) -> str:
    q = question.lower().strip()

    # ─── Hóa đơn / Đơn hàng ───
    if any(kw in q for kw in ['hóa đơn', 'hoa don', 'đơn hàng', 'don hang', 'invoice', 'order']):
        if any(kw in q for kw in ['số', 'bao nhiêu', 'tổng', 'total', 'count']):
            total = df_all['InvoiceNo'].nunique()
            return f"Tổng số hóa đơn (unique): **{total:,}**\nĐây là số hóa đơn không trùng lặp trong toàn bộ dataset."
        if any(kw in q for kw in ['trung bình', 'average', 'avg', 'tb']):
            avg_val = df_all.groupby('InvoiceNo')['Revenue'].sum().mean()
            return f"Giá trị trung bình mỗi hóa đơn: **£{avg_val:,.0f}**"
        total = df_all['InvoiceNo'].nunique()
        return f"Tổng số hóa đơn: **{total:,}**"

    # ─── Doanh thu ───
    if any(kw in q for kw in ['doanh thu', 'revenue', 'doanh số', 'doanh so']):
        total = df_all['Revenue'].sum()
        if any(kw in q for kw in ['tháng', 'thang', 'month']):
            monthly = df_all.groupby(df_all['InvoiceDate'].dt.to_period('M').astype(str))['Revenue'].sum()
            result = f"Tổng doanh thu: **£{total:,.0f}**\n\nDoanh thu theo tháng:\n"
            for m, rev in monthly.items():
                result += f"- {m}: £{rev:,.0f}\n"
            peak = monthly.idxmax()
            result += f"\nTháng cao nhất: **{peak}** (£{monthly.max():,.0f})"
            return result
        if any(kw in q for kw in ['ngày', 'ngay', 'day']):
            day_rev = df_all.groupby('DayOfWeek')['Revenue'].sum()
            result = "Doanh thu theo ngày trong tuần:\n"
            for d, rev in day_rev.items():
                result += f"- {d}: £{rev:,.0f}\n"
            return result
        peak_month = df_all.groupby(df_all['InvoiceDate'].dt.to_period('M').astype(str))['Revenue'].sum().idxmax()
        return f"Tổng doanh thu: **£{total:,.0f}**\nTháng đỉnh: **{peak_month}**"

    # ─── Khách hàng ───
    if any(kw in q for kw in ['khách hàng', 'khach hang', 'customer', 'người mua']):
        if any(kw in q for kw in ['số', 'bao nhiêu', 'tổng', 'count']):
            total = rfm['CustomerID'].nunique()
            return f"Tổng số khách hàng (có tài khoản): **{total:,}**"
        if any(kw in q for kw in ['top', 'cao nhất', 'nhiều nhất', 'vip', 'best']):
            top_cust = rfm.nlargest(5, 'Monetary')[['CustomerID', 'Monetary', 'Frequency', 'Segment']]
            result = "Top 5 khách hàng chi tiêu cao nhất:\n"
            for _, row in top_cust.iterrows():
                result += f"- KH #{int(row['CustomerID'])}: £{row['Monetary']:,.0f} ({row['Segment']}, {int(row['Frequency'])} lần mua)\n"
            return result
        total = rfm['CustomerID'].nunique()
        segments = rfm['Segment'].value_counts()
        result = f"Tổng khách hàng: **{total:,}**\n\nPhân nhóm:\n"
        for seg, cnt in segments.items():
            result += f"- {seg}: {cnt:,} ({cnt/total*100:.1f}%)\n"
        return result

    # ─── Sản phẩm ───
    if any(kw in q for kw in ['sản phẩm', 'san pham', 'product', 'mặt hàng', 'mat hang', 'item']):
        if any(kw in q for kw in ['số', 'bao nhiêu', 'tổng', 'count']):
            total = df_all['StockCode'].nunique()
            return f"Tổng số sản phẩm (StockCode unique): **{total:,}**"
        if any(kw in q for kw in ['top', 'cao nhất', 'bán chạy', 'best']):
            top = df_all.groupby('Description')['Revenue'].sum().nlargest(10)
            result = "Top 10 sản phẩm doanh thu cao nhất:\n"
            for i, (name, rev) in enumerate(top.items(), 1):
                result += f"{i}. {name}: £{rev:,.0f}\n"
            return result
        total = df_all['StockCode'].nunique()
        return f"Tổng số sản phẩm: **{total:,}**"

    # ─── Quốc gia ───
    if any(kw in q for kw in ['quốc gia', 'quoc gia', 'country', 'nước', 'nuoc']):
        top = df_all.groupby('Country')['Revenue'].sum().nlargest(10)
        result = "Top 10 quốc gia theo doanh thu:\n"
        for i, (name, rev) in enumerate(top.items(), 1):
            result += f"{i}. {name}: £{rev:,.0f}\n"
        result += f"\nTổng số quốc gia: **{df_all['Country'].nunique()}**"
        return result

    # ─── Churn / Rời bỏ ───
    if any(kw in q for kw in ['churn', 'rời bỏ', 'roi bo', 'risk', 'nguy cơ', 'nguy co']):
        high_risk = rfm[rfm['ChurnProba'] > 0.7]
        medium_risk = rfm[(rfm['ChurnProba'] > 0.4) & (rfm['ChurnProba'] <= 0.7)]
        result = f"**Phân tích nguy cơ churn:**\n"
        result += f"- Nguy cơ cao (>70%): **{len(high_risk):,}** khách hàng\n"
        result += f"- Nguy cơ trung bình (40-70%): **{len(medium_risk):,}** khách hàng\n"
        result += f"- Tỷ lệ churn trung bình: **{rfm['ChurnProba'].mean()*100:.1f}%**\n"
        result += f"\nNhóm chiếm đa số: **{rfm['Segment'].value_counts().index[0]}**"
        return result

    # ─── RFM / Phân nhóm ───
    if any(kw in q for kw in ['rfm', 'segment', 'phân nhóm', 'phan nhom', 'nhóm', 'nhom', 'cluster']):
        segments = rfm['Segment'].value_counts()
        result = "Phân nhóm khách hàng RFMD:\n"
        for seg, cnt in segments.items():
            avg_monetary = rfm[rfm['Segment'] == seg]['Monetary'].mean()
            result += f"- **{seg}**: {cnt:,} khách (chi tiêu TB: £{avg_monetary:,.0f})\n"
        return result

    # ─── Association Rules / Luật kết hợp ───
    if any(kw in q for kw in ['association', 'luật', 'luat', 'kết hợp', 'ket hop', 'combo', 'bundle', 'basket']):
        if rules_df is not None and len(rules_df) > 0:
            top3 = rules_df.nlargest(3, 'lift')
            result = "Top 3 luật kết hợp mạnh nhất (Market Basket):\n"
            for _, row in top3.iterrows():
                result += f"- Nếu mua **{row['antecedents']}** → thường mua **{row['consequents']}** (Confidence: {row['confidence']*100:.0f}%, Lift: {row['lift']:.1f})\n"
            result += f"\nTổng: **{len(rules_df)}** luật kết hợp"
            return result
        return "Chưa có dữ liệu Association Rules."

    # ─── Dự báo / Forecast ───
    if any(kw in q for kw in ['dự báo', 'du bao', 'forecast', 'predict', 'tương lai', 'tuong lai']):
        if forecast_summary is not None:
            result = "Kết quả so sánh mô hình dự báo:\n"
            for _, row in forecast_summary.iterrows():
                result += f"- **{row['Algorithm']}**: {row['Best Approach']} — MAPE: {row['Best MAPE (%)']}%\n"
            result += "\n(MAPE càng thấp = dự báo càng chính xác)"
            return result
        return "Chưa có dữ liệu dự báo. Hãy chạy predictive-modeling.py trước."

    # ─── Giá trị / Thống kê cơ bản ───
    if any(kw in q for kw in ['trung bình', 'average', 'mean', 'tb']):
        if any(kw in q for kw in ['giá', 'gia', 'price']):
            avg_price = df_all['UnitPrice'].mean()
            return f"Đơn giá trung bình: **£{avg_price:.2f}**"
        avg_rev = df_all['Revenue'].mean()
        return f"Doanh thu trung bình mỗi giao dịch: **£{avg_rev:.2f}**"

    # ─── Tổng quan nhanh ───
    if any(kw in q for kw in ['tổng quan', 'tong quan', 'overview', 'summary', 'tóm tắt', 'tom tat']):
        total_revenue = df_all['Revenue'].sum()
        total_orders = df_all['InvoiceNo'].nunique()
        total_customers = rfm['CustomerID'].nunique()
        total_products = df_all['StockCode'].nunique()
        return (f"**Tổng quan Dashboard:**\n"
                f"- Tổng doanh thu: £{total_revenue:,.0f}\n"
                f"- Số hóa đơn: {total_orders:,}\n"
                f"- Số khách hàng: {total_customers:,}\n"
                f"- Số sản phẩm: {total_products:,}\n"
                f"- Giai đoạn: {df_all['InvoiceDate'].min().date()} → {df_all['InvoiceDate'].max().date()}")

    # ─── Các câu hỏi "là gì" / "là bao nhiêu" — catch-all ───
    if 'là' in q:
        keywords_map = {
            'doanh thu': ('Tổng doanh thu', f"£{df_all['Revenue'].sum():,.0f}"),
            'hóa đơn': ('Số hóa đơn', f"{df_all['InvoiceNo'].nunique():,}"),
            'hoa don': ('Số hóa đơn', f"{df_all['InvoiceNo'].nunique():,}"),
            'khách hàng': ('Số khách hàng', f"{rfm['CustomerID'].nunique():,}"),
            'khach hang': ('Số khách hàng', f"{rfm['CustomerID'].nunique():,}"),
            'sản phẩm': ('Số sản phẩm', f"{df_all['StockCode'].nunique():,}"),
            'san pham': ('Số sản phẩm', f"{df_all['StockCode'].nunique():,}"),
            'quốc gia': ('Số quốc gia', f"{df_all['Country'].nunique()}"),
            'quoc gia': ('Số quốc gia', f"{df_all['Country'].nunique()}"),
        }
        for kw, (label, value) in keywords_map.items():
            if kw in q:
                return f"{label}: **{value}**"

    # ─── Fallback ───
    return ("Tôi chưa hiểu rõ câu hỏi. Bạn có thể hỏi về:\n"
            "- **Doanh thu**: tổng doanh thu, doanh thu theo tháng/ngày\n"
            "- **Hóa đơn**: số hóa đơn, giá trị trung bình\n"
            "- **Khách hàng**: số khách, top khách hàng, phân nhóm RFM\n"
            "- **Sản phẩm**: số sản phẩm, top sản phẩm\n"
            "- **Quốc gia**: top quốc gia theo doanh thu\n"
            "- **Churn**: nguy cơ rời bỏ, tỷ lệ churn\n"
            "- **Luật kết hợp**: combo sản phẩm, cross-sell\n"
            "- **Dự báo**: kết quả forecast\n"
            "- **Tổng quan**: tóm tắt dashboard")


with tab5:
    st.markdown('<div class="section-header">🤖 Hỏi đáp Business Intelligence</div>', unsafe_allow_html=True)
    st.caption("Hỏi bất kỳ câu hỏi nào về dữ liệu — hệ thống sẽ truy vấn và trả lời bằng tiếng Việt hoặc tiếng Anh")

    with st.expander("💡 Gợi ý câu hỏi"):
        st.markdown("""
| Chủ đề | Ví dụ câu hỏi |
|--------|---------------|
| Doanh thu | "Tổng doanh thu là bao nhiêu?", "Doanh thu theo tháng?" |
| Hóa đơn | "Số hóa đơn là?", "Giá trị trung bình mỗi đơn?" |
| Khách hàng | "Top 5 khách hàng VIP?", "Phân nhóm khách hàng?" |
| Sản phẩm | "Top 10 sản phẩm bán chạy?", "Có bao nhiêu sản phẩm?" |
| Churn | "Bao nhiêu khách có nguy cơ churn?", "Tỷ lệ churn?" |
| Association | "Combo sản phẩm phổ biến nhất?", "Luật kết hợp?" |
| Dự báo | "Kết quả dự báo doanh thu?" |
| Tổng quan | "Tóm tắt dashboard", "Overview" |
        """)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    user_input = st.chat_input("Ví dụ: Số hóa đơn là bao nhiêu? / Top 5 sản phẩm? / Tỷ lệ churn?")

    if user_input:
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            with st.spinner('Đang phân tích dữ liệu...'):
                response = get_ai_response(user_input, df_all, rfm)
                st.markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
