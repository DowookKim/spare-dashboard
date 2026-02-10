import streamlit as st

from data_loader import load_excel_data, list_item_sheets, load_item_inventory
from data_processor import (
    preprocess_spare_data,
    preprocess_summary_metrics,
    add_inventory_status,
    compute_kpis,
)
from visualizer import create_spare_dashboard, create_metrics_dashboard, style_inventory_table


st.set_page_config(page_title="SPARE 대시보드", layout="wide")
st.title("📊 SPARE 대시보드")

uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])

if uploaded_file is None:
    st.info("엑셀 파일을 업로드하면 대시보드가 표시됩니다.")
    st.stop()

# 기존 로딩(기존 2개 페이지에서만 사용)
raw_df, sheet_name = load_excel_data(uploaded_file)

# 기존 페이지용 전처리
category_df = preprocess_spare_data(raw_df)
metric_long = preprocess_summary_metrics(uploaded_file, sheet_name)

st.success(f"시트 로드 완료: {sheet_name}")

# -----------------------------
# Sidebar: 페이지 + 공통 설정
# -----------------------------
page = st.sidebar.radio(
    "분석 페이지 선택",
    ["부품 카테고리별 분석", "전체 집계 지표", "아이템 재고 상태"],
)

chart_style = st.sidebar.radio("차트 스타일", ["막대그래프", "꺾은선"], index=0)

# -----------------------------
# Page 1) 부품 카테고리별 분석 (변경 금지: 기존 로직 유지)
# -----------------------------
if page == "부품 카테고리별 분석":
    # 기존(카테고리별) 대시보드는 Category 선택값이 필요함
    categories = (
        category_df["Category"].dropna().astype(str).unique().tolist()
        if "Category" in category_df.columns
        else []
    )

    if not categories:
        st.warning("카테고리(Category) 컬럼을 찾지 못해 그래프를 생성할 수 없습니다.")
        st.stop()

    selected_category = st.sidebar.selectbox("부품 카테고리", categories, index=0)

    fig = create_spare_dashboard(category_df, selected_category, chart_style)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Page 2) 전체 집계 지표 (변경 금지: 기존 로직 유지)
# -----------------------------
elif page == "전체 집계 지표":
    fig = create_metrics_dashboard(metric_long, chart_style)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Page 3) 아이템 재고 상태 (신규)
# -----------------------------
else:
    st.header("아이템 재고 상태 대시보드")

    # 1) 시트 목록 로딩
    item_sheets = list_item_sheets(uploaded_file, spare_sheet_name=sheet_name)
    if not item_sheets:
        st.warning("SPARE 현황을 제외한 아이템 시트를 찾지 못했습니다.")
        st.stop()

    # 2) 사이드바 컨트롤 (새 페이지에서만)
    selected_sheet = st.sidebar.selectbox("아이템(시트)", item_sheets, index=0)
    only_risky = st.sidebar.checkbox("품절/위험만 보기", value=False)
    top_n = st.sidebar.slider("Top N(위험 우선순위)", min_value=5, max_value=100, value=20, step=1)

    # 3) 아이템 시트 로딩 + 2단 헤더 디버그 포함
    inv_df, dbg = load_item_inventory(uploaded_file, selected_sheet, return_debug=True)

    if dbg.get("error"):
        st.error(dbg["error"])

    if inv_df.empty:
        st.warning("해당 시트에서 재고 데이터를 추출하지 못했습니다.")

        with st.expander("디버그: 헤더 탐지/컬럼 매핑 정보"):
            st.json(dbg)
        st.stop()

    # 4) 상태 판정 + KPI
    inv_status = add_inventory_status(inv_df)
    kpis = compute_kpis(inv_status)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 품목 수", f"{kpis['total']:,}")
    c2.metric("위험 품목 수", f"{kpis['risk']:,}")
    c3.metric("위험 비율", f"{kpis['risk_rate']:.1f}%")
    c4.metric("품절 수", f"{kpis['stockout']:,}")

    # 5) 위험/품절 Top N
    st.subheader("🚨 위험/품절 Top N")

    risk_df = inv_status[inv_status["상태"].isin(["품절", "위험"])].copy()
    risk_df = risk_df.sort_values(["부족량", "현재고"], ascending=[False, True])

    top_cols = ["대분류", "중분류", "소분류", "품명", "규격", "안전재고", "현재고", "부족량", "상태"]
    risk_top = risk_df[top_cols].head(top_n)

    if risk_top.empty:
        st.info("현재 선택된 아이템 시트에서 위험/품절 품목이 없습니다.")
    else:
        st.dataframe(risk_top, use_container_width=True)

        # CSV 다운로드
        csv_bytes = risk_top.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSV 다운로드(Top N)",
            data=csv_bytes,
            file_name=f"{selected_sheet}_risk_top{top_n}.csv",
            mime="text/csv",
        )

    # 6) 전체 리스트(조건부 스타일)
    st.subheader("📋 전체 품목 리스트")

    show_df = inv_status.copy()

    # 상태 아이콘(표시용)
    icon_map = {"품절": "⛔", "위험": "🟥", "안전": "🟩", "기준없음": "⚪"}
    status_raw = show_df["상태"].copy()
    show_df["상태"] = status_raw.map(lambda s: f"{icon_map.get(s, '⚪')} {s}")

    if only_risky:
        show_df = show_df[status_raw.isin(["품절", "위험"])].copy()
        status_raw = status_raw.loc[show_df.index]

    # 표시 컬럼(최소)
    all_cols = ["대분류", "중분류", "소분류", "품명", "규격", "안전재고", "현재고", "부족량", "충족률", "상태"]
    show_df = show_df[all_cols]

    styled = style_inventory_table(show_df, status_raw=status_raw)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # 7) 디버그 정보
    with st.expander("디버그: 헤더 탐지/컬럼 매핑 정보"):
        st.write("선택 시트:", selected_sheet)
        st.json(dbg)
