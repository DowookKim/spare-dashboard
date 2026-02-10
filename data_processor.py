import pandas as pd
from typing import Dict

def preprocess_spare_data(df):
    """
    데이터 정제, 시간순 정렬 및 수치 데이터 안전 변환 모듈
    '탈거품' 등 비수치 문자열에 대한 예외 처리 포함
    TOTAL 행의 재고비용, 재고수량도 포함
    """
    # 1. 컬럼 인덱스 정의
    date_col_name = df.columns[0]
    metric_col_name = df.columns[1]

    # 2. 날짜 결측치 처리 및 날짜 객체 변환 (정렬용)
    df[date_col_name] = df[date_col_name].ffill()
    
    # '2024년 12월' -> datetime 객체 변환 (실패 시 NaT)
    df['Date_dt'] = pd.to_datetime(df[date_col_name], format='%Y년 %m월', errors='coerce')

    # 3. 메트릭 필터링 강화 (부품 카테고리별 분석용)
    # 분석 대상인 "항목수", "재고비용", "재고수량"만 포함된 행만 남김
    df['Metric_Clean'] = df[metric_col_name].astype(str).str.replace(" ", "")
    target_metrics = ["항목수", "재고비용", "재고수량", "재고수량(EA)"]
    
    # TOTAL 행 처리: 첫 번째 컬럼이 "TOTAL"이고 메트릭이 있는 경우도 포함
    df_filtered = df[
        (df['Metric_Clean'].isin(target_metrics)) | 
        ((df[date_col_name].astype(str).str.upper() == "TOTAL") & 
         (df['Metric_Clean'].isin(target_metrics)))
    ].copy()

    # 4. 시간순 정렬 (2024년 12월부터 오름차순)
    df_filtered = df_filtered.sort_values('Date_dt')

    # 5. Wide-to-Long 데이터 재구조화 (Melt)
    # 날짜와 메트릭을 제외한 모든 컬럼을 'Category'로 통합
    id_vars = [date_col_name, 'Date_dt', 'Metric_Clean']
    # 비고, Unnamed, 합계 등 제외할 컬럼 패턴 정의
    exclude_patterns = ["비고", "Unnamed", "합계", "총계", "누계"]
    val_vars = [c for c in df_filtered.columns if c not in id_vars and c != metric_col_name 
                and not any(p in c for p in exclude_patterns)]
    
    df_melted = df_filtered.melt(id_vars=id_vars, value_vars=val_vars, 
                         var_name='Category', value_name='Value')

    # 6. 안전한 수치 변환 로직 (ValueError 해결 핵심)
    # [1] 문자열 변환 및 콤마 제거
    df_melted['Value'] = df_melted['Value'].astype(str).str.replace(',', '').str.strip()
    
    # [2] pd.to_numeric 활용: 숫자가 아닌 '탈거품', '-', 공백 등은 모두 NaN으로 강제 변환(coerce)
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
    
    # [3] NaN 값을 0.0으로 채우고 데이터 타입을 float로 확정
    df_melted['Value'] = df_melted['Value'].fillna(0.0).astype(float)

    # 7. 최종 컬럼명 정리 및 반환
    df_melted.rename(columns={
        date_col_name: 'Date_Str', 
        'Date_dt': 'Date', 
        'Metric_Clean': 'Metric'
    }, inplace=True)
    
    # 재고수량(EA) -> 재고수량으로 통일
    df_melted['Metric'] = df_melted['Metric'].str.replace('재고수량(EA)', '재고수량')

    return df_melted


def preprocess_summary_metrics(uploaded_file, sheet_name):
    """
    집계 지표 데이터 전처리 (Excel 42~51행)
    - 원본 파일을 header=None으로 직접 읽어서 처리
    - 42행: 월 헤더
    - 43~51행: 각 메트릭 데이터
    """
    import pandas as pd
    
    # 원본 데이터를 header=None으로 읽기
    df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
    
    # Excel 42행 (인덱스 41) - 월 헤더
    month_row = df_raw.iloc[41]
    
    # 월 컬럼 추출 (인덱스 2부터 시작)
    months = []
    for i in range(2, len(month_row)):
        val = month_row[i]
        if pd.notna(val) and '년' in str(val) and '월' in str(val):
            months.append((i, str(val).strip()))
    
    # 메트릭 행 매핑 (Excel 43~51행 = 인덱스 42~50)
    metric_rows = {
        42: '항목수',
        43: '재고비용',
        44: '재고수량',
        45: '설비가대비재고보유율',
        46: '스페어구매비용',
        47: '스페어재고대비구매비율',
        48: '구매비용전월대비증감율',
        49: '스페어미보유수량',
        50: '스페어미보유율'
    }
    
    # 데이터 추출
    result_data = []
    for row_idx, metric_name in metric_rows.items():
        if row_idx >= len(df_raw):
            continue
        row_data = df_raw.iloc[row_idx]
        
        for col_idx, month_name in months:
            value = row_data[col_idx]
            result_data.append({
                'Metric': metric_name,
                'Month': month_name,
                'Value': value
            })
    
    df_result = pd.DataFrame(result_data)
    
    # 수치 변환
    df_result['Value'] = df_result['Value'].astype(str).str.replace(',', '').str.strip()
    df_result['Value'] = pd.to_numeric(df_result['Value'], errors='coerce')
    df_result['Value'] = df_result['Value'].fillna(0.0).astype(float)
    
    # 월 순서 정렬
    def month_sort_key(month_str):
        if '2024년 12월' in month_str:
            return 202412
        if '2025년' in month_str:
            try:
                month_num = int(month_str.split('년')[1].replace('월', '').strip())
                return 202500 + month_num
            except:
                return 999999
        return 999999
    
    df_result['Month_Sort'] = df_result['Month'].apply(month_sort_key)
    df_result = df_result.sort_values('Month_Sort')
    df_result['Month_Clean'] = df_result['Month']
    
    return df_result

# -----------------------------
# 아이템(=시트) 재고 상태 페이지용
# -----------------------------

def add_inventory_status(df: pd.DataFrame) -> pd.DataFrame:
    """안전재고/현재고 기준으로 상태, 부족량, 충족률을 계산해 컬럼 추가.

    상태 정의(고정):
      - 품절: 현재고 == 0
      - 위험: 0 < 현재고 < 안전재고
      - 안전: 현재고 >= 안전재고
      - 기준없음: 안전재고가 NaN 또는 0, 또는 현재고가 NaN

    주의: 여기서의 연산은 '표현/판정'을 위한 최소한의 계산만 수행한다.
    """
    out = df.copy()

    # 숫자화(원본이 이미 숫자여도 안전)
    out["안전재고"] = pd.to_numeric(out.get("안전재고"), errors="coerce")
    out["현재고"] = pd.to_numeric(out.get("현재고"), errors="coerce")

    safe = out["안전재고"]
    curr = out["현재고"]

    safe_valid = safe.notna() & (safe > 0)
    curr_valid = curr.notna()

    status = pd.Series(["기준없음"] * len(out), index=out.index)

    # 품절/위험/안전은 안전재고가 유효하고 현재고가 숫자로 존재할 때만 판정
    mask = safe_valid & curr_valid

    status.loc[mask & (curr == 0)] = "품절"
    status.loc[mask & (curr > 0) & (curr < safe)] = "위험"
    status.loc[mask & (curr >= safe)] = "안전"

    out["상태"] = status

    # 부족량 / 충족률
    out["부족량"] = pd.NA
    out.loc[mask, "부족량"] = (safe.loc[mask] - curr.loc[mask]).clip(lower=0)

    out["충족률"] = pd.NA
    out.loc[mask, "충족률"] = (curr.loc[mask] / safe.loc[mask]).replace([pd.NA, pd.NaT], pd.NA)

    return out


def compute_kpis(df_with_status: pd.DataFrame) -> Dict[str, float]:
    """상단 KPI(전체/위험/위험비율/품절)를 계산."""
    total = int(len(df_with_status))

    status = df_with_status.get("상태")
    if status is None:
        return {
            "total": total,
            "risk": 0,
            "risk_rate": 0.0,
            "stockout": 0,
        }

    stockout = int((status == "품절").sum())
    risk = int((status == "위험").sum())

    # 위험비율 분모: 기준없음 제외
    denom = int(status.isin(["품절", "위험", "안전"]).sum())
    risk_rate = float(risk / denom) if denom > 0 else 0.0

    return {
        "total": total,
        "risk": risk,
        "risk_rate": risk_rate,
        "stockout": stockout,
    }
