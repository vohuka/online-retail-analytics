# app.py — Dashboard chính
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="Retail BI Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ===== LOAD DỮ LIỆU =====
@st.cache_data  # Cache để không load lại mỗi lần click
def load_data():
    df_all      = pd.read_csv('outputs/df_all_clean.csv', parse_dates=['InvoiceDate'])
    df_customer = pd.read_csv('outputs/df_customer_clean.csv', parse_dates=['InvoiceDate'])
    rfm         = pd.read_csv('outputs/rfm_with_predictions.csv')
    return df_all, df_customer, rfm

df_all, df_customer, rfm = load_data()

# ===== TIÊU ĐỀ =====
st.title("🛒 Online Retail — Business Intelligence Dashboard")
st.caption("Dataset: UCI Online Retail | Dec 2010 – Dec 2011 | UK-based giftware retailer")
st.divider()

# Thêm đoạn này vào đầu file app.py, trước phần tab

import os

def get_ai_response(question: str, df_all: pd.DataFrame, rfm: pd.DataFrame) -> str:
    """
    Nhận câu hỏi tự nhiên → dùng AI generate Python code → chạy → trả kết quả
    """
    # Tóm tắt schema dữ liệu để AI hiểu
    context = f"""
Bạn là một Business Intelligence analyst. Bạn có 2 dataframe sau:

1. df_all: {len(df_all):,} giao dịch với các cột:
   - InvoiceNo (mã hóa đơn), StockCode (mã sản phẩm), Description (tên sản phẩm)
   - Quantity (số lượng), UnitPrice (đơn giá £), Revenue (doanh thu = Quantity × UnitPrice)
   - InvoiceDate (ngày giao dịch), Country (quốc gia khách)
   - Month (1-12), DayOfWeek (Monday-Sunday), Hour (0-23)

2. rfm: {len(rfm):,} khách hàng với các cột:
   - CustomerID, Recency (ngày từ lần mua cuối), Frequency (số lần mua)
   - Monetary (tổng chi tiêu £), Diversity (số sản phẩm khác nhau)
   - Segment (nhóm: Champions/Loyal Customers/At Risk/...)
   - ChurnProba (xác suất churn 0.0-1.0)

Câu hỏi: {question}

Hãy trả lời bằng tiếng Việt. Nếu cần số liệu, hãy mô tả kết quả rõ ràng.
Không được tự bịa số liệu. Nếu không rõ câu hỏi, hãy hỏi lại.
"""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": context}]
        )
        return message.content[0].text
    except Exception as e:
        # Fallback: trả lời bằng code Python thuần nếu không có API
        return simple_query(question, df_all, rfm)


def simple_query(question: str, df_all, rfm) -> str:
    """Fallback không cần API — xử lý các câu hỏi phổ biến"""
    q = question.lower()
    if 'top' in q and 'product' in q:
        top = df_all.groupby('Description')['Revenue'].sum().nlargest(5)
        result = "Top 5 sản phẩm doanh thu cao nhất:\n"
        for i, (name, rev) in enumerate(top.items(), 1):
            result += f"{i}. {name}: £{rev:,.0f}\n"
        return result
    elif 'churn' in q or 'risk' in q:
        at_risk = rfm[rfm['ChurnProba'] > 0.7]
        return f"Có {len(at_risk):,} khách hàng có nguy cơ churn cao (xác suất > 70%).\nNhóm nguy hiểm nhất: {rfm['Segment'].value_counts().index[0]}"
    elif 'revenue' in q or 'doanh thu' in q:
        total = df_all['Revenue'].sum()
        peak_month = df_all.groupby('Month')['Revenue'].sum().idxmax()
        return f"Tổng doanh thu: £{total:,.0f}\nTháng đỉnh: Tháng {peak_month}"
    else:
        return "Tôi chưa hiểu câu hỏi này. Bạn có thể hỏi về: doanh thu, sản phẩm, khách hàng, churn risk."

# ===== TAB LAYOUT =====
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan",
    "👥 Khách hàng",
    "📈 Dự báo",
    "🤖 Chatbot BI"
])

# ===== TAB 1: TỔNG QUAN =====
with tab1:
    # Hàng KPI metrics
    col1, col2, col3, col4 = st.columns(4)

    total_revenue   = df_all['Revenue'].sum()
    total_orders    = df_all['InvoiceNo'].nunique()
    total_customers = df_customer['CustomerID'].nunique()
    total_products  = df_all['StockCode'].nunique()

    col1.metric("💰 Tổng doanh thu",    f"£{total_revenue:,.0f}")
    col2.metric("📦 Số hóa đơn",        f"{total_orders:,}")
    col3.metric("👤 Khách có tài khoản", f"{total_customers:,}")
    col4.metric("🏷️ Sản phẩm unique",   f"{total_products:,}")

    st.divider()

    # Biểu đồ doanh thu theo tháng
    monthly = df_all.groupby(df_all['InvoiceDate'].dt.to_period('M').astype(str))['Revenue'].sum().reset_index()
    monthly.columns = ['Tháng', 'Doanh thu']

    fig_monthly = px.line(monthly, x='Tháng', y='Doanh thu',
                          title='Doanh thu theo tháng',
                          markers=True,
                          color_discrete_sequence=['#1976D2'])
    fig_monthly.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Top 10 sản phẩm
    col_left, col_right = st.columns(2)

    with col_left:
        top_products = df_all.groupby('Description')['Revenue'].sum().nlargest(10).reset_index()
        fig_prod = px.bar(top_products.sort_values('Revenue'),
                          x='Revenue', y='Description',
                          orientation='h',
                          title='Top 10 sản phẩm theo doanh thu',
                          color_discrete_sequence=['#43A047'])
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_right:
        top_countries = df_all.groupby('Country')['Revenue'].sum().nlargest(10).reset_index()
        fig_country = px.bar(top_countries.sort_values('Revenue'),
                             x='Revenue', y='Country',
                             orientation='h',
                             title='Top 10 quốc gia theo doanh thu',
                             color_discrete_sequence=['#F4511E'])
        st.plotly_chart(fig_country, use_container_width=True)

# ===== TAB 2: KHÁCH HÀNG =====
with tab2:
    st.subheader("Phân nhóm khách hàng theo RFMD Clustering")

    # Biểu đồ donut phân nhóm
    segment_counts = rfm['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Nhóm', 'Số khách']

    col_a, col_b = st.columns(2)
    with col_a:
        fig_donut = px.pie(segment_counts, values='Số khách', names='Nhóm',
                           hole=0.4, title='Phân bố nhóm khách hàng')
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        # Scatter plot: Frequency vs Monetary, màu theo nhóm
        fig_scatter = px.scatter(rfm, x='Frequency', y='Monetary',
                                 color='Segment', size='Diversity',
                                 hover_data=['CustomerID'],
                                 title='Biểu đồ RFM — Frequency vs Monetary',
                                 opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Bảng khách hàng có nguy cơ churn cao nhất
    st.subheader("⚠️ Top 20 khách hàng có nguy cơ churn cao nhất")
    churn_risk = rfm.nlargest(20, 'ChurnProba')[
        ['CustomerID','Recency','Frequency','Monetary','Segment','ChurnProba']
    ]
    churn_risk['ChurnProba'] = (churn_risk['ChurnProba'] * 100).round(1).astype(str) + '%'
    st.dataframe(churn_risk, use_container_width=True, hide_index=True)

# ===== TAB 3: DỰ BÁO =====
with tab3:
    st.subheader("📈 Dự báo doanh thu")
    st.info("Mô hình Prophet đã được huấn luyện trên dữ liệu Jan–Oct 2011, dự báo Nov–Dec 2011")

    # Load và hiển thị kết quả forecast đã lưu
    try:
        forecast_result = pd.read_csv('outputs/forecast_results.csv', parse_dates=['ds'])
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast_result['ds'], y=forecast_result['actual'],
            name='Thực tế', line=dict(color='#1976D2', width=2)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_result['ds'], y=forecast_result['predicted'],
            name='Dự báo', line=dict(color='#E53935', width=2, dash='dash')
        ))
        fig_forecast.update_layout(title='Dự báo doanh thu vs Thực tế')
        st.plotly_chart(fig_forecast, use_container_width=True)
    except:
        st.warning("Chưa có file forecast_results.csv — chạy Phase 4 trước")

# ===== TAB 4: CHATBOT BI =====
with tab4:
    st.subheader("🤖 Hỏi đáp Business Intelligence")
    st.caption("Gõ câu hỏi bằng tiếng Anh hoặc tiếng Việt — hệ thống sẽ truy vấn dữ liệu và trả lời")

    # Lịch sử chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # Input của người dùng
    user_input = st.chat_input("Ví dụ: What are the top 5 products? / Khách hàng nào mua nhiều nhất?")

    if user_input:
        # Thêm câu hỏi vào lịch sử
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.write(user_input)

        # Gọi AI để trả lời (phần này tích hợp API ở bước 5.2)
        with st.chat_message('assistant'):
            with st.spinner('Đang phân tích...'):
                response = get_ai_response(user_input, df_all, rfm)
                st.write(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
