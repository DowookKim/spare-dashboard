import plotly.graph_objects as go
from typing import Dict
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def create_spare_dashboard(df, selected_category, chart_style):
    """3단 서브플롯 대시보드 생성 (부품 카테고리별)"""
    plot_df = df[df['Category'] == selected_category].copy()
    plot_df = plot_df.sort_values('Date')

    # 정렬된 날짜 순서를 명시적으로 추출 (X축 순서 고정용)
    date_order = plot_df['Date_Str'].drop_duplicates().tolist()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"📦 {selected_category} 항목수", f"💰 {selected_category} 재고비용", f"🔢 {selected_category} 재고수량")
    )

    metrics = [("항목수", 1, "#636EFA"), ("재고비용", 2, "#EF553B"), ("재고수량", 3, "#00CC96")]

    for m_name, row_idx, color in metrics:
        data = plot_df[plot_df['Metric'] == m_name]
        
        if "막대" in chart_style:
            trace = go.Bar(x=data['Date_Str'], y=data['Value'], name=m_name, marker_color=color)
        else:
            trace = go.Scatter(x=data['Date_Str'], y=data['Value'], mode='lines+markers', name=m_name, line=dict(color=color, width=3))
        fig.add_trace(trace, row=row_idx, col=1)
        
        # [핵심] 모든 행의 X축 라벨을 보이게 설정
        fig.update_xaxes(showticklabels=True, row=row_idx, col=1, tickangle=45)

    fig.update_layout(height=900, showlegend=False, hovermode="x unified", margin=dict(t=100, b=50))
    fig.update_yaxes(tickformat=",") # 천단위 콤마
    
    return fig


def create_metrics_dashboard(df, chart_style):
    """
    9개 집계 지표를 한 화면에 표시하는 대시보드
    - 9x1 서브플롯 레이아웃 (세로 배치)
    - 전월 대비 증감 표시 (색상)
    - 최대/최소값 마커
    """
    
    # 메트릭 정의 및 레이아웃 (9x1 세로 배치)
    metrics_info = [
        ('항목수', '📦 전체 항목수', 1, 1, False),
        ('재고비용', '💰 총 재고비용', 2, 1, False),
        ('재고수량', '🔢 총 재고수량', 3, 1, False),
        ('설비가대비재고보유율', '📊 설비가 대비 재고 보유율', 4, 1, True),
        ('스페어구매비용', '💳 스페어 구매비용', 5, 1, False),
        ('스페어재고대비구매비율', '📈 스페어 재고대비 구매 비율', 6, 1, True),
        ('구매비용전월대비증감율', '📉 구매비용 전월대비 증감율', 7, 1, True),
        ('스페어미보유수량', '⚠️ 스페어 미보유 수량', 8, 1, False),
        ('스페어미보유율', '⚡ 스페어 미보유율', 9, 1, True)
    ]
    
    fig = make_subplots(
        rows=9, cols=1,
        subplot_titles=[info[1] for info in metrics_info],
        vertical_spacing=0.08,
        shared_xaxes=False
    )
    
    for metric_name, title, row, col, is_percentage in metrics_info:
        # 데이터 필터링 및 정렬
        metric_df = df[df['Metric'] == metric_name].copy()
        metric_df = metric_df.sort_values('Month_Sort')
        
        if len(metric_df) == 0:
            continue
        
        # X축은 Month 컬럼 직접 사용 (이미 "2024년 12월" 형식)
        x_vals = metric_df['Month'].tolist()
        y_vals = metric_df['Value'].tolist()
        
        # 전월 대비 증감 계산
        color_vals = []
        for i in range(len(y_vals)):
            if i == 0:
                color_vals.append('gray')
            else:
                if y_vals[i] > y_vals[i-1]:
                    color_vals.append('green')
                elif y_vals[i] < y_vals[i-1]:
                    color_vals.append('red')
                else:
                    color_vals.append('gray')
        
        # 차트 타입 선택
        if "막대" in chart_style:
            trace = go.Bar(
                x=x_vals, 
                y=y_vals, 
                name=metric_name,
                marker=dict(
                    color=color_vals,
                    line=dict(color='white', width=1)
                ),
                showlegend=False
            )
        else:
            # 라인 차트에서는 전월 대비 증감을 마커 색상으로 표시
            trace = go.Scatter(
                x=x_vals, 
                y=y_vals, 
                mode='lines+markers',
                name=metric_name,
                line=dict(color='#636EFA', width=2),
                marker=dict(
                    size=10,
                    color=color_vals,
                    line=dict(color='white', width=1)
                ),
                showlegend=False
            )
        
        fig.add_trace(trace, row=row, col=col)
        
        # 최대/최소값 마커 추가
        if len(y_vals) > 0:
            max_idx = np.argmax(y_vals)
            min_idx = np.argmin(y_vals)
            
            # 최대값 마커
            fig.add_trace(
                go.Scatter(
                    x=[x_vals[max_idx]],
                    y=[y_vals[max_idx]],
                    mode='markers+text',
                    marker=dict(size=15, color='gold', symbol='star', line=dict(color='orange', width=2)),
                    text=['MAX'],
                    textposition='top center',
                    showlegend=False,
                    hovertemplate=f'최대: {y_vals[max_idx]:,.2f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # 최소값 마커
            fig.add_trace(
                go.Scatter(
                    x=[x_vals[min_idx]],
                    y=[y_vals[min_idx]],
                    mode='markers+text',
                    marker=dict(size=15, color='lightblue', symbol='diamond', line=dict(color='blue', width=2)),
                    text=['MIN'],
                    textposition='bottom center',
                    showlegend=False,
                    hovertemplate=f'최소: {y_vals[min_idx]:,.2f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Y축 포맷 설정
        if is_percentage:
            fig.update_yaxes(tickformat=".2%", row=row, col=col)
        else:
            fig.update_yaxes(tickformat=",", row=row, col=col)
        
        # X축 설정
        fig.update_xaxes(showticklabels=True, tickangle=45, row=row, col=col)
    
    fig.update_layout(
        height=2400,
        showlegend=False,
        hovermode="closest",
        margin=dict(t=80, b=60, l=60, r=60),
        title_text="📊 전체 집계 지표 대시보드",
        title_x=0.5,
        title_font_size=20
    )
    
    return fig

# -----------------------------
# 아이템(=시트) 재고 상태 테이블 스타일
# -----------------------------

def style_inventory_table(df: pd.DataFrame, status_raw: pd.Series | None = None) -> "pd.io.formats.style.Styler":
    """Streamlit st.dataframe에 넣을 Styler 반환.

    - 행 배경색: 품절/위험/안전/기준없음
    - 충족률: bar

    status_raw를 따로 받는 이유:
      df['상태']에 아이콘을 붙여 표시할 때, 원본 상태값으로 색상을 정확히 칠하기 위해서.
    """
    if status_raw is None:
        status_raw = df.get("상태")

    # CSS 컬러는 너무 튀지 않게, 하지만 구분은 되도록
    row_colors = {
        "품절": "background-color: rgba(255, 0, 0, 0.12);",
        "위험": "background-color: rgba(255, 165, 0, 0.14);",
        "안전": "background-color: rgba(0, 128, 0, 0.10);",
        "기준없음": "background-color: rgba(0, 0, 0, 0.04);",
    }

    def _style_row(i: int) -> list[str]:
        try:
            s = status_raw.iloc[i]
        except Exception:
            s = None
        css = row_colors.get(str(s), "")
        return [css] * len(df.columns)

    styler = df.style
    styler = styler.apply(lambda _row: _style_row(_row.name), axis=1)

    # 포맷
    fmt: Dict[str, str] = {}
    if "안전재고" in df.columns:
        fmt["안전재고"] = "{:,.0f}"
    if "현재고" in df.columns:
        fmt["현재고"] = "{:,.0f}"
    if "부족량" in df.columns:
        fmt["부족량"] = "{:,.0f}"
    if "충족률" in df.columns:
        fmt["충족률"] = "{:.0%}"

    if fmt:
        styler = styler.format(fmt, na_rep="")

    # databar
    if "충족률" in df.columns:
        try:
            styler = styler.bar(subset=["충족률"], vmin=0, vmax=1)
        except Exception:
            pass

    return styler
