"""Phân tích hành vi khách hàng đăng ký SIM và lựa chọn gói cước Viettel."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "khach_hang_viettel.xlsx"
logger = logging.getLogger(__name__)
SHEET_NAME = "Du_lieu_khach_hang"
RED = "#E60023"
BLUE = "#2563EB"
COLORS = ["#E60023", "#293D63", "#FF8A56", "#46A29F", "#8A78E6", "#F1B44C"]

C = {
    "id": "Customer_ID",
    "age": "Tuổi",
    "gender": "Giới tính",
    "region": "Khu vực",
    "date": "Ngày đăng ký SIM",
    "sim": "Loại SIM",
    "plan": "Gói cước đăng ký",
    "price": "Giá gói cước (VNĐ)",
    "duration": "Thời hạn gói (ngày)",
    "data": "Data sử dụng (GB)",
    "minutes": "Số phút gọi",
    "sms": "Số SMS",
    "renewals": "Số lần gia hạn",
    "revenue": "Doanh thu khách hàng (VNĐ)",
    "status": "Trạng thái",
}
NUMERIC_COLUMNS = [C[k] for k in ("age", "price", "duration", "data", "minutes", "sms", "renewals", "revenue")]

st.set_page_config(
    page_title="Viettel | Phân tích hành vi khách hàng",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { font-family: "Segoe UI", Arial, sans-serif; background: #fff; }
    .block-container { max-width: 1360px; padding: 2rem 2.5rem 3rem; }
    #MainMenu { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    .v-hero {
        background: #fff; color: #17283D;
        border: 0;
        border-radius: 0; padding: 16px 12px 28px; margin-bottom: 20px;
        box-shadow: none; text-align: center;
    }
    .v-hero .eyebrow { font-size: .75rem; font-weight: 600; letter-spacing: .12em; color: #526075; margin-bottom: 14px; }
    .v-hero .v-data-tag { display: block; padding: 0; margin: 12px 0 0; border-radius: 0; background: transparent; color: #66758A; font-size: .75rem; }
    .v-hero h1 { color: #17283D !important; max-width: 1040px; margin: 0 auto 10px; padding: 0; font-family: "Times New Roman", Georgia, serif; font-size: 2.1rem; line-height: 1.35; font-weight: 700; letter-spacing: 0; }
    .v-hero > p { margin: 0; color: #526075; font-size: 1rem; }
    h2, h3, h4 { color: #17283D; font-family: "Times New Roman", Georgia, serif; letter-spacing: 0; }
    h3 { font-size: 1.25rem !important; padding-top: .4rem !important; }
    h4 { font-size: 1rem !important; }
    [role="tablist"] { gap: 0; padding: 0; background: #fff; border: 1px solid #C9D1DB; border-radius: 0; overflow-x: auto; max-width: 100%; }
    [role="tablist"] [role="tab"] { flex: 0 0 auto; border-radius: 0; border-right: 1px solid #E0E4E9; border-bottom: 2px solid transparent; padding: 11px 18px; font-weight: 600; color: #526075; }
    [role="tablist"] [role="tab"] p { margin: 0; font-size: .85rem; color: inherit; }
    [role="tablist"] [role="tab"][aria-selected="true"] { color: #BA1530; background: #FFF3F5; border-bottom-color: #BA1530; }
    [role="tablist"] .react-aria-SelectionIndicator { display: none; }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }
    [data-testid="stTabPanel"] { padding-top: 14px; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #C9D1DB;
        padding: 18px; border-radius: 0; min-height: 112px;
        box-shadow: none;
    }
    [data-testid="stMetricLabel"] p { color: #63738A; font-size: .8rem; font-weight: 500; }
    [data-testid="stMetricValue"] { color: #17283D; font-size: clamp(1.2rem, 2vw, 1.7rem); font-weight: 700; letter-spacing: -.035em; }
    [data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #C9D1DB; border-radius: 0; padding: 6px; overflow: hidden; }
    [data-testid="stDataFrame"] { background: #fff; border: 1px solid #C9D1DB; border-radius: 0; overflow: hidden; }
    .stApp :is(button, input, textarea, [role="combobox"]) { border-radius: 0 !important; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #66758A !important; font-size: .78rem; opacity: 1; }
    [data-testid="stWidgetLabel"] p { font-size: .83rem; font-weight: 500; }
    .section-note { color: #66758A; font-size: .78rem; margin: 6px 0 16px; }
    @media (max-width: 760px) {
        .block-container { padding: 1.25rem 1rem 2rem; }
        .v-hero { padding: 12px 0 24px; }
        .v-hero h1 { font-size: 1.65rem; }
        .v-hero .eyebrow { font-size: .68rem; letter-spacing: .07em; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def vnd(value: float) -> str:
    """Định dạng tiền Việt, không làm tròn về triệu khi xem dữ liệu chi tiết."""
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}".replace(",", ".") + " ₫"


def chart_style(fig, height=365):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=47, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#334155", size=12),
        title_font=dict(size=16, color="#142338"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EFF2F7", zeroline=False)
    return fig


def plot(fig):
    st.plotly_chart(chart_style(fig), width="stretch", config={"displaylogo": False})


@st.cache_data(show_spinner="Đang đọc dữ liệu khách hàng...")
def load_data(path: str, mtime: float) -> pd.DataFrame:
    """Cache theo thời điểm cập nhật file: Excel đổi thì dữ liệu tự nạp lại."""
    _ = mtime
    df = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl", dtype={C["id"]: str})
    df.columns = df.columns.astype(str).str.strip()
    missing = [col for col in C.values() if col not in df.columns]
    if missing:
        raise ValueError("File thiếu các cột bắt buộc: " + ", ".join(missing))

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tệp Excel có thể lưu ngày dưới dạng datetime hoặc serial Excel.
    raw_dates = df[C["date"]]
    if pd.api.types.is_numeric_dtype(raw_dates):
        df[C["date"]] = pd.to_datetime(raw_dates, unit="D", origin="1899-12-30", errors="coerce")
    else:
        df[C["date"]] = pd.to_datetime(raw_dates, errors="coerce", dayfirst=True)

    df[C["id"]] = df[C["id"]].astype("string").str.strip()
    for key in ("gender", "region", "sim", "plan", "status"):
        col = C[key]
        df[col] = df[col].astype("string").str.strip().fillna("Không rõ")
    return df


@st.cache_data(show_spinner="Đang phân cụm khách hàng...")
def kmeans_analysis(df: pd.DataFrame, k: int):
    features = [C[x] for x in ("age", "price", "data", "minutes", "sms", "renewals", "revenue")]
    valid = df.dropna(subset=features).copy()
    if len(valid) <= k:
        return None, None, None
    scaler = StandardScaler()
    X = scaler.fit_transform(valid[features])
    if len(np.unique(X, axis=0)) < k:
        return None, None, None
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    valid["Cụm"] = [f"Cụm {int(i) + 1}" for i in labels]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    valid["Thành phần 1"] = coords[:, 0]
    valid["Thành phần 2"] = coords[:, 1]
    score = float(silhouette_score(X, labels)) if k < len(valid) else np.nan
    return valid, features, score


if not DATA_FILE.exists():
    st.error("Chưa tìm thấy tệp dữ liệu khach_hang_viettel.xlsx.")
    st.stop()

try:
    data = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as exc:
    logger.exception("Không đọc được dữ liệu khách hàng: %s", type(exc).__name__)
    st.error("Chưa đọc được dữ liệu khách hàng. Vui lòng kiểm tra trang tính Du_lieu_khach_hang và các cột dữ liệu.")
    st.stop()

st.markdown(
    """
    <div class="v-hero">
      <div class="eyebrow">BÁO CÁO PHÂN TÍCH DỮ LIỆU</div>
      <h1>Phân tích hành vi khách hàng Viettel</h1>
      <p>Đăng ký SIM và lựa chọn gói cước di động</p>
      <p class="v-data-tag">Dữ liệu mô phỏng</p>
    </div>
    """,
    unsafe_allow_html=True,
)
# ----- Bộ lọc chung: tác động lên tất cả các màn hình -----
st.markdown("#### Phạm vi phân tích")
f1, f2, f3, f4 = st.columns([1.15, 1.2, 1, 1.05])
with f1:
    selected_regions = st.multiselect("Khu vực", sorted(data[C["region"]].dropna().unique()), placeholder="Tất cả khu vực")
with f2:
    selected_plans = st.multiselect("Gói cước", sorted(data[C["plan"]].dropna().unique()), placeholder="Tất cả gói cước")
with f3:
    selected_sims = st.multiselect("Loại SIM", sorted(data[C["sim"]].dropna().unique()), placeholder="Tất cả loại SIM")
with f4:
    selected_statuses = st.multiselect("Trạng thái", sorted(data[C["status"]].dropna().unique()), placeholder="Tất cả trạng thái")

df = data.copy()
for col, values in ((C["region"], selected_regions), (C["plan"], selected_plans),
                    (C["sim"], selected_sims), (C["status"], selected_statuses)):
    if values:
        df = df.loc[df[col].isin(values)]

st.caption(f"**{len(df):,} / {len(data):,}** khách hàng · **{df[C['plan']].nunique()}** gói cước · **{df[C['region']].nunique()}** khu vực")
if df.empty:
    st.warning("Không có khách hàng phù hợp các điều kiện lọc. Hãy bỏ bớt một số lựa chọn.")
    st.stop()

# Thanh menu NẰM NGANG, không dùng sidebar và không có chức năng tải lên dữ liệu.
tabs = st.tabs([
    "Tổng quan", "Khách hàng", "Gói cước & sử dụng",
    "Chi tiết đăng ký", "Phân khúc khách hàng", "Chất lượng dữ liệu",
])

with tabs[0]:
    st.subheader("Tổng quan khách hàng")
    customers = len(df)
    active_count = int((df[C["status"]] == "Đang sử dụng").sum())
    total_revenue = df[C["revenue"]].sum()
    top_plan = df[C["plan"]].value_counts().index[0]
    metrics = st.columns(5)
    metrics[0].metric("Khách hàng", f"{customers:,}")
    metrics[1].metric("Đang sử dụng", f"{active_count:,}", help=f"Chiếm {active_count/customers:.1%} số khách hàng đang phân tích.")
    metrics[2].metric("Tổng doanh thu ghi nhận", vnd(total_revenue))
    metrics[3].metric("Số gói cước", f"{df[C['plan']].nunique():,}")
    metrics[4].metric("Gói phổ biến nhất", str(top_plan))
    st.markdown('<div class="section-note">Doanh thu ghi nhận: tổng doanh thu khách hàng trong phạm vi phân tích.</div>', unsafe_allow_html=True)

    a, b = st.columns([1.55, 1])
    with a:
        rank = df[C["plan"]].value_counts().reset_index()
        rank.columns = ["Gói cước", "Khách hàng"]
        fig = px.bar(rank, x="Gói cước", y="Khách hàng", color_discrete_sequence=[RED],
                     title="Số khách hàng đăng ký theo gói cước", text="Khách hàng")
        fig.update_traces(textposition="outside", cliponaxis=False)
        plot(fig)
    with b:
        status = df[C["status"]].value_counts().reset_index()
        status.columns = ["Trạng thái", "Số lượng"]
        fig = px.pie(status, names="Trạng thái", values="Số lượng", hole=.62,
                     color_discrete_sequence=["#11A583", RED], title="Trạng thái sử dụng SIM")
        fig.update_traces(textinfo="percent+label")
        plot(fig)

    a, b = st.columns([1.1, 1])
    with a:
        timeline = df.dropna(subset=[C["date"]]).copy()
        if not timeline.empty:
            timeline["Tháng"] = timeline[C["date"]].dt.to_period("M").dt.to_timestamp()
            monthly = timeline.groupby("Tháng").size().reset_index(name="Lượt đăng ký")
            fig = px.line(monthly, x="Tháng", y="Lượt đăng ký", markers=True,
                          color_discrete_sequence=[RED], title="Số lượt đăng ký SIM theo tháng")
            fig.update_traces(line=dict(width=3))
            plot(fig)
        else:
            st.info("Chưa có dữ liệu ngày đăng ký hợp lệ để hiển thị biểu đồ.")
    with b:
        regions = df[C["region"]].value_counts().reset_index()
        regions.columns = ["Khu vực", "Khách hàng"]
        fig = px.bar(regions, x="Khách hàng", y="Khu vực", orientation="h",
                     color_discrete_sequence=[BLUE], title="Khách hàng theo khu vực")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        plot(fig)

with tabs[1]:
    st.subheader("Thống kê đặc điểm khách hàng")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tuổi trung bình", f"{df[C['age']].mean():.1f}")
    c2.metric("Nam", f"{(df[C['gender']] == 'Nam').sum():,}")
    c3.metric("Nữ", f"{(df[C['gender']] == 'Nữ').sum():,}")
    c4.metric("Trả trước / Trả sau", f"{(df[C['sim']] == 'Trả trước').sum():,} / {(df[C['sim']] == 'Trả sau').sum():,}")

    a, b = st.columns(2)
    with a:
        gender = df[C["gender"]].value_counts().reset_index()
        gender.columns = ["Giới tính", "Số khách hàng"]
        fig = px.pie(gender, names="Giới tính", values="Số khách hàng", hole=.35,
                     color_discrete_sequence=[RED, BLUE], title="Phân bố theo giới tính")
        plot(fig)
    with b:
        age = df.copy()
        age["Nhóm tuổi"] = pd.cut(age[C["age"]], bins=[0, 17, 24, 34, 44, 54, np.inf],
                                 labels=["<18", "18–24", "25–34", "35–44", "45–54", "55+"])
        age_group = age.groupby("Nhóm tuổi", observed=False).size().reset_index(name="Khách hàng")
        fig = px.bar(age_group, x="Nhóm tuổi", y="Khách hàng", title="Cơ cấu khách hàng theo nhóm tuổi",
                     color_discrete_sequence=[RED], text="Khách hàng")
        plot(fig)
    a, b = st.columns(2)
    with a:
        sims = df[C["sim"]].value_counts().reset_index()
        sims.columns = ["Loại SIM", "Khách hàng"]
        fig = px.bar(sims, x="Loại SIM", y="Khách hàng", title="Khách hàng sử dụng trả trước / trả sau",
                     color="Loại SIM", color_discrete_sequence=COLORS, text="Khách hàng")
        plot(fig)
    with b:
        loc_status = df.groupby([C["region"], C["status"]]).size().reset_index(name="Khách hàng")
        fig = px.bar(loc_status, x=C["region"], y="Khách hàng", color=C["status"],
                     barmode="stack", color_discrete_sequence=["#17A682", RED],
                     title="Trạng thái khách hàng theo khu vực")
        plot(fig)

with tabs[2]:
    st.subheader("Gói cước đã đăng ký và hành vi sử dụng")
    by_plan = (
        df.groupby(C["plan"], dropna=False)
        .agg(
            **{"Số khách hàng": (C["id"], "count"),
               "Giá gói TB (VNĐ)": (C["price"], "mean"),
               "Data TB (GB)": (C["data"], "mean"),
               "Phút gọi TB": (C["minutes"], "mean"),
               "Gia hạn TB": (C["renewals"], "mean"),
               "Tổng doanh thu ghi nhận (VNĐ)": (C["revenue"], "sum")}
        )
        .reset_index().sort_values("Số khách hàng", ascending=False)
    )
    st.dataframe(by_plan, hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        fig = px.bar(by_plan, x=C["plan"], y="Tổng doanh thu ghi nhận (VNĐ)",
                     title="Tổng doanh thu ghi nhận theo gói cước", color_discrete_sequence=[RED])
        plot(fig)
    with b:
        fig = px.bar(by_plan, x=C["plan"], y="Data TB (GB)",
                     title="Dung lượng data sử dụng trung bình theo gói", color_discrete_sequence=[BLUE])
        plot(fig)

    st.markdown("#### Cơ cấu gói cước theo khu vực")
    cross = pd.crosstab(df[C["region"]], df[C["plan"]])
    fig = go.Figure(data=go.Heatmap(
        x=list(cross.columns), y=list(cross.index), z=cross.values,
        colorscale=[[0, "#FFF1F2"], [.5, "#F46C7D"], [1, "#B80029"]],
        text=cross.values, texttemplate="%{text}",
        hovertemplate="Khu vực: %{y}<br>Gói: %{x}<br>Khách: %{z}<extra></extra>",
    ))
    fig.update_layout(title="Ma trận số khách hàng theo khu vực × gói cước", height=440)
    plot(fig)

    st.markdown("#### Chi tiết theo gói cước")
    plan_choice = st.selectbox("Chọn gói để xem chi tiết", sorted(df[C["plan"]].unique()), key="plan_details")
    one_plan = df[df[C["plan"]] == plan_choice]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Số khách hàng", f"{len(one_plan):,}")
    p2.metric("Tỷ trọng trong bộ lọc", f"{len(one_plan)/len(df):.1%}")
    p3.metric("Data TB", f"{one_plan[C['data']].mean():.2f} GB")
    p4.metric("Doanh thu ghi nhận", vnd(one_plan[C["revenue"]].sum()))
    st.dataframe(one_plan[[C["id"], C["region"], C["sim"], C["plan"], C["status"]]],
                 width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Chi tiết khách hàng và gói cước")
    st.caption("Tra cứu theo mã khách hàng, gói cước hoặc khu vực.")
    q1, q2 = st.columns([2, 1])
    with q1:
        query = st.text_input("Tìm kiếm", placeholder="Ví dụ: VT00001, ST70K, Hà Nội...")
    with q2:
        age_min = int(df[C["age"]].min()) if df[C["age"]].notna().any() else 0
        age_max = int(df[C["age"]].max()) if df[C["age"]].notna().any() else 100
        if age_min == age_max:
            age_limit = (age_min, age_max)
            st.caption(f"Tất cả khách hàng đang lọc đều {age_min} tuổi")
        else:
            age_limit = st.slider("Độ tuổi", min_value=age_min, max_value=age_max,
                                  value=(age_min, age_max))
    records = df.loc[df[C["age"]].between(*age_limit)].copy()
    if query.strip():
        q = query.strip()
        match = (records[C["id"]].str.contains(q, case=False, na=False, regex=False)
                 | records[C["plan"]].str.contains(q, case=False, na=False, regex=False)
                 | records[C["region"]].str.contains(q, case=False, na=False, regex=False))
        records = records.loc[match]

    st.metric("Kết quả tra cứu", f"{len(records):,}")
    show_cols = [C[k] for k in ("id", "age", "gender", "region", "date", "sim", "plan",
                                 "price", "data", "renewals", "revenue", "status")]
    display = records[show_cols].copy()
    display[C["date"]] = display[C["date"]].dt.strftime("%d/%m/%Y").fillna("—")
    st.dataframe(display, width="stretch", hide_index=True, height=420)
    st.download_button(
        "Xuất danh sách đã lọc (CSV)",
        display.to_csv(index=False).encode("utf-8-sig"),
        file_name="danh_sach_khach_hang_viettel.csv", mime="text/csv",
    )

    if not records.empty:
        selected_id = st.selectbox("Xem thông tin chi tiết khách hàng", ["-- Chọn mã khách hàng --"] +
                                   sorted(records[C["id"]].dropna().astype(str).unique().tolist()))
        if selected_id != "-- Chọn mã khách hàng --":
            r = records.loc[records[C["id"]].astype(str) == selected_id].iloc[0]
            with st.container(border=True):
                st.markdown(f"**Khách hàng {selected_id}**")
                l, m, n = st.columns(3)
                l.write(f"**Tuổi:** {r[C['age']]:.0f}  |  **Giới tính:** {r[C['gender']]}")
                l.write(f"**Khu vực:** {r[C['region']]}")
                m.write(f"**Loại SIM:** {r[C['sim']]}")
                m.write(f"**Đăng ký gói:** {r[C['plan']]}")
                n.write(f"**Trạng thái:** {r[C['status']]}")
                n.write(f"**Giá gói:** {vnd(r[C['price']])}")
                st.caption(f"Ngày đăng ký: {r[C['date']].strftime('%d/%m/%Y') if pd.notna(r[C['date']]) else 'Không rõ'}"
                           f"  ·  Data đã sử dụng: {r[C['data']]:.2f} GB"
                           f"  ·  Số lần gia hạn: {r[C['renewals']]:.0f}")

with tabs[4]:
    st.subheader("Phân khúc khách hàng bằng K-Means")
    st.caption("Phân nhóm theo tuổi, giá gói, mức sử dụng, số lần gia hạn và doanh thu; các đặc trưng được chuẩn hóa trước khi phân cụm.")
    max_k = min(6, max(2, len(df) - 1))
    if len(df) < 4:
        st.warning("Cần ít nhất 4 khách hàng sau lọc để chạy K-Means.")
    else:
        k = st.slider("Số cụm K", 2, max_k, 2, key="kmeans_k")
        cluster_df, features, score = kmeans_analysis(df, k)
        if cluster_df is None:
            st.warning("Không đủ bản ghi hợp lệ để phân cụm.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Khách hàng được phân cụm", len(cluster_df))
            m2.metric("Số cụm", k)
            m3.metric("Silhouette score", f"{score:.3f}" if pd.notna(score) else "—")
            fig = px.scatter(
                cluster_df, x="Thành phần 1", y="Thành phần 2", color="Cụm",
                hover_name=C["id"], hover_data=[C["plan"], C["region"]],
                color_discrete_sequence=COLORS, opacity=.72,
                title=f"Bản đồ phân cụm {k} nhóm khách hàng (PCA)",
            )
            plot(fig)
            cluster_summary = (
                cluster_df.groupby("Cụm")
                .agg(
                    **{"Số khách hàng": (C["id"], "count"),
                       "Tuổi TB": (C["age"], "mean"),
                       "Giá gói TB (VNĐ)": (C["price"], "mean"),
                       "Data TB (GB)": (C["data"], "mean"),
                       "Gia hạn TB": (C["renewals"], "mean"),
                       "Doanh thu TB (VNĐ)": (C["revenue"], "mean")}
                ).round(2).reset_index()
            )
            st.markdown("#### Đặc trưng trung bình của từng cụm")
            st.dataframe(cluster_summary, width="stretch", hide_index=True)
            cluster_plans = cluster_df.groupby(["Cụm", C["plan"]]).size().reset_index(name="Khách hàng")
            fig = px.bar(cluster_plans, x="Cụm", y="Khách hàng", color=C["plan"],
                         barmode="stack", color_discrete_sequence=px.colors.qualitative.Safe,
                         title="Các gói cước được chọn trong từng cụm")
            plot(fig)
with tabs[5]:
    st.subheader("Chất lượng dữ liệu khách hàng")
    st.caption(f"Đánh giá trên toàn bộ {len(data):,} bản ghi của bộ dữ liệu gốc.")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Số dòng", f"{len(data):,}")
    q2.metric("Số cột", len(data.columns))
    q3.metric("Ô thiếu dữ liệu", f"{int(data.isna().sum().sum()):,}")
    q4.metric("ID trùng", f"{int(data[C['id']].duplicated().sum()):,}")
    quality = pd.DataFrame({
        "Trường dữ liệu": data.columns,
        "Số ô thiếu": data.isna().sum().values,
        "Tỷ lệ thiếu (%)": (data.isna().mean() * 100).round(2).values,
        "Kiểu dữ liệu": data.dtypes.astype(str).values,
    })
    st.dataframe(quality, hide_index=True, width="stretch")
