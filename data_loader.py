import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _DummyStreamlit:
        def cache_data(self, *args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator

    st = _DummyStreamlit()


# -----------------------------
# (기존) SPARE 현황 시트 로딩
# -----------------------------

def get_sheet_name(uploaded_file):
    """엑셀 파일에서 SPARE 현황과 유사한 시트 이름을 찾음"""
    xls = pd.ExcelFile(uploaded_file)
    for sheet in xls.sheet_names:
        if "SPARE" in sheet.upper().replace(" ", ""):
            return sheet
    return xls.sheet_names[0]  # 못 찾으면 첫 번째 시트 반환


def load_excel_data(uploaded_file):
    """데이터 로드 및 기본적인 컬럼 정리 (기존 2개 페이지에서 사용)

    주의: 기존 페이지 로직을 유지하기 위해 signature/동작을 바꾸지 않는다.
    """
    target_sheet = get_sheet_name(uploaded_file)

    # 두 번째 행(header=1)부터 읽고 52행까지 확장 (TOTAL 행 및 집계 지표 포함)
    df = pd.read_excel(uploaded_file, sheet_name=target_sheet, header=1, nrows=52)

    # 컬럼명 앞뒤 공백 및 줄바꿈 제거
    df.columns = [str(col).replace("\n", " ").strip() for col in df.columns]
    return df, target_sheet


# -----------------------------
# (신규) 아이템(=시트) 재고 상태 페이지용 로딩
# -----------------------------

def _to_bytes(uploaded_or_bytes: Union[bytes, "st.runtime.uploaded_file_manager.UploadedFile"]):
    if isinstance(uploaded_or_bytes, (bytes, bytearray)):
        return bytes(uploaded_or_bytes)
    return uploaded_or_bytes.getvalue()


def list_item_sheets(uploaded_or_bytes, spare_sheet_name: Optional[str] = None) -> List[str]:
    """엑셀 시트 목록 중 SPARE 현황 시트를 제외한 나머지 시트를 반환."""
    file_bytes = _to_bytes(uploaded_or_bytes)
    xls = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")

    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", str(s or "").upper())

    spare_norm = _norm(spare_sheet_name) if spare_sheet_name else None

    out = []
    for name in xls.sheet_names:
        n = _norm(name)
        if spare_norm and n == spare_norm:
            continue
        # 혹시 sheet_name을 못 넘겨도 SPARE 현황 성격의 시트는 제외
        if "SPARE" in n and "현황" in n:
            continue
        out.append(name)
    return out


def _is_blank(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    s = str(v).strip()
    if s == "":
        return True
    if s.lower() == "nan":
        return True
    return False


def _clean_header_token(v) -> str:
    if _is_blank(v):
        return ""
    s = str(v)
    s = s.replace("\n", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # 엑셀에서 Unnamed: 형태가 들어오는 경우가 있는데, 헤더 토큰으로는 빈값 취급
    if s.upper().startswith("UNNAMED"):
        return ""
    return s


def _norm(s) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", "", s)
    return s.upper()


def _score_header_pair(upper_row: List, lower_row: List) -> Tuple[int, Dict[str, bool]]:
    """2단 헤더 후보 (upper, lower) 조합 스코어링."""

    upper_join = _norm(" ".join(["" if _is_blank(x) else str(x) for x in upper_row]))
    lower_join = _norm(" ".join(["" if _is_blank(x) else str(x) for x in lower_row]))

    groups = {
        "safety": ["안전", "MIN", "최소"],
        "current": ["현재"],
        "name": ["품명", "품목", "자재명"],
        "spec": ["규격", "사양", "SPEC"],
        "cat": ["대분류", "중분류", "소분류"],
    }

    hits = {}
    for g, keys in groups.items():
        hit_u = any(k in upper_join for k in keys)
        hit_l = any(k in lower_join for k in keys)
        hits[g] = hit_u or hit_l

    hit_count = sum(bool(v) for v in hits.values())

    # upper에 안전/현재가 있으면 더 신뢰
    bonus = 0
    if any(k in upper_join for k in groups["safety"]):
        bonus += 8
    if any(k in upper_join for k in groups["current"]):
        bonus += 6

    score = hit_count * 10 + bonus
    return score, hits


def _build_df_from_two_level_header(
    raw: pd.DataFrame,
    upper_idx: int,
    lower_idx: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]], List[str], List[str]]:
    """raw에서 (upper_idx, lower_idx)를 2단 헤더로 결합하여 df를 생성.

    Returns
    -------
    df : DataFrame
    col_meta : {col_name: {"upper": ..., "lower": ...}}
    upper_ffill_preview : upper header ffill 결과
    lower_preview : lower header 리스트
    """

    upper_raw = raw.iloc[upper_idx].tolist()
    lower_raw = raw.iloc[lower_idx].tolist()

    # upper ffill (병합셀 대응)
    upper_ffill: List[str] = []
    last = ""
    for v in upper_raw:
        tok = _clean_header_token(v)
        if tok == "" and last != "":
            tok = last
        if tok != "":
            last = tok
        upper_ffill.append(tok)

    lower_clean = [_clean_header_token(v) for v in lower_raw]

    cols: List[str] = []
    meta: Dict[str, Dict[str, str]] = {}
    used = {}

    for i, (u, l) in enumerate(zip(upper_ffill, lower_clean)):
        if u == "" and l == "":
            name = f"__EMPTY_{i}__"
        elif u == "":
            name = l
        elif l == "" or _norm(l) == _norm(u):
            name = u
        else:
            name = f"{u}_{l}"

        # 정규화 (중복 underscore/공백)
        name = name.replace("\xa0", " ").replace("\n", " ").strip()
        name = re.sub(r"\s+", " ", name)
        name = re.sub(r"_+", "_", name)

        # 중복 방지
        base = name
        if base in used:
            used[base] += 1
            name = f"{base}_{used[base]}"
        else:
            used[base] = 1

        cols.append(name)
        meta[name] = {"upper": u, "lower": l}

    df = raw.iloc[lower_idx :].copy()
    df.columns = cols

    # 완전 빈 컬럼 제거: 헤더도 비고 값도 전부 NaN
    drop_cols = []
    for c in df.columns:
        if c.startswith("__EMPTY_") and df[c].isna().all():
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
        for c in drop_cols:
            meta.pop(c, None)

    return df, meta, upper_ffill, lower_clean


def _numeric_success_rate(s: pd.Series, sample: int = 200) -> float:
    if s is None:
        return 0.0
    ss = s.head(sample)
    # 문자열로 변환 후 콤마 제거
    ss = ss.astype(str).str.replace(",", "", regex=False)
    ss = ss.str.replace("\xa0", " ", regex=False).str.strip()
    num = pd.to_numeric(ss, errors="coerce")
    if len(num) == 0:
        return 0.0
    return float(num.notna().mean())


def _pick_best_column(
    df: pd.DataFrame,
    col_meta: Dict[str, Dict[str, str]],
    candidates: List[str],
    prefer_lower_contains: Optional[List[str]] = None,
) -> Optional[str]:
    if not candidates:
        return None

    prefer_lower_contains = prefer_lower_contains or []

    best = None
    best_score = -1e9

    for c in candidates:
        lower = col_meta.get(c, {}).get("lower", "")
        lower_n = _norm(lower)
        score = 0.0

        # lower에 '수량'이 있으면 우선
        if any(_norm(k) in lower_n for k in prefer_lower_contains):
            score += 2.0

        # 숫자 변환 성공률(결측률이 낮을수록 가산)
        score += _numeric_success_rate(df[c])

        # 안전한 tie-breaker: 값이 더 많이 채워진 것을 우선
        try:
            score += float(df[c].notna().mean()) * 0.5
        except Exception:
            pass

        if score > best_score:
            best_score = score
            best = c

    return best


def _pick_text_column(df: pd.DataFrame, synonyms: List[str]) -> Optional[str]:
    syn_norm = [_norm(x) for x in synonyms]

    cands = []
    for c in df.columns:
        cn = _norm(c)
        if any(s in cn for s in syn_norm):
            cands.append(c)

    if not cands:
        return None

    best = None
    best_score = -1
    for c in cands:
        s = df[c]
        # 텍스트 컬럼은 빈 문자열/NaN이 적을수록 좋음
        try:
            filled = s.astype(str).str.strip()
            score = float((filled != "").mean())
        except Exception:
            score = float(s.notna().mean())
        if score > best_score:
            best_score = score
            best = c

    return best


@st.cache_data(show_spinner=False)
def load_item_inventory(
    uploaded_or_bytes,
    sheet_name: str,
    return_debug: bool = False,
    max_scan_rows: int = 40,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """아이템 시트에서 재고 테이블을 표준 스키마로 추출.

    - 2단 헤더(병합셀 포함)를 우선 처리
    - 안전재고/현재고가 서로 다른 열로 매핑되도록 강제
    """

    file_bytes = _to_bytes(uploaded_or_bytes)
    raw = pd.read_excel(
        BytesIO(file_bytes),
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
        dtype=object,
    )

    debug: Dict = {
        "sheet_name": sheet_name,
        "mode": "two_level",
        "chosen_upper_idx": None,
        "chosen_lower_idx": None,
        "upper_ffill_preview": [],
        "lower_preview": [],
        "combined_columns": [],
        "mapped_columns": {},
        "candidates": {},
        "error": None,
    }

    n_rows = raw.shape[0]
    scan_rows = min(max_scan_rows, n_rows)

    # 1) 2단 헤더 후보 (upper_idx, lower_idx) 조합 점수화
    pair_scores: List[Tuple[int, int, int]] = []
    pair_hits: Dict[Tuple[int, int], Dict[str, bool]] = {}

    for u in range(0, max(0, scan_rows - 1)):
        upper = raw.iloc[u].tolist()
        for l in range(u + 1, min(scan_rows, u + 4)):
            lower = raw.iloc[l].tolist()
            score, hits = _score_header_pair(upper, lower)
            if sum(bool(v) for v in hits.values()) >= 3:
                pair_scores.append((score, u, l))
                pair_hits[(u, l)] = hits

    pair_scores.sort(key=lambda x: (-x[0], x[1], x[2]))

    # 후보가 너무 많으면 상위만
    pair_scores = pair_scores[:25]

    # 2) 각 후보로 df 생성 → 안전재고/현재고 매핑 성공하는 첫 번째 조합 채택
    chosen_df: Optional[pd.DataFrame] = None
    chosen_meta: Optional[Dict[str, Dict[str, str]]] = None
    chosen_upper: Optional[List[str]] = None
    chosen_lower: Optional[List[str]] = None
    chosen_u = None
    chosen_l = None
    chosen_safe = None
    chosen_curr = None

    def _try_map(df: pd.DataFrame, meta: Dict[str, Dict[str, str]]):
        # 안전재고 후보: 안전/MIN/최소 포함
        safety_keys = ["안전", "MIN", "최소"]
        current_keys = ["현재"]

        safety_cands = [c for c in df.columns if any(k in _norm(c) for k in safety_keys)]
        current_cands = [c for c in df.columns if any(k in _norm(c) for k in current_keys)]

        # 현재고가 '현재'로 안 잡히면 fallback: '재고' 포함이되 안전키는 제외
        if not current_cands:
            current_cands = [
                c
                for c in df.columns
                if ("재고" in _norm(c)) and not any(k in _norm(c) for k in safety_keys)
            ]

        debug["candidates"] = {
            "safety": safety_cands,
            "current": current_cands,
        }

        safe_col = _pick_best_column(df, meta, safety_cands, prefer_lower_contains=["수량"])
        # 현재고는 안전재고와 같은 열을 선택하면 안 됨 → safe_col 제외
        curr_cands_ex = [c for c in current_cands if c != safe_col]
        curr_col = _pick_best_column(df, meta, curr_cands_ex, prefer_lower_contains=["수량"])

        return safe_col, curr_col

    if pair_scores:
        for score, u, l in pair_scores:
            df_tmp, meta_tmp, upper_ffill, lower_list = _build_df_from_two_level_header(raw, u, l)
            safe_col, curr_col = _try_map(df_tmp, meta_tmp)

            if safe_col is None or curr_col is None:
                continue
            if safe_col == curr_col:
                continue

            chosen_df, chosen_meta = df_tmp, meta_tmp
            chosen_upper, chosen_lower = upper_ffill, lower_list
            chosen_u, chosen_l = u, l
            chosen_safe, chosen_curr = safe_col, curr_col
            break

    # 3) 2단 헤더로 실패하면: 단일 헤더로라도 시도 (지원 범위)
    if chosen_df is None:
        debug["mode"] = "single_level_fallback"

        # 단일 헤더 후보 행 찾기: 키워드 3개 이상 포함
        best = None
        best_score = -1
        for r in range(scan_rows):
            row = raw.iloc[r].tolist()
            row_join = _norm(" ".join(["" if _is_blank(x) else str(x) for x in row]))
            keys = ["안전", "MIN", "최소", "현재", "품명", "품목", "규격", "대분류", "중분류", "소분류"]
            score = sum(k in row_join for k in keys)
            if score >= 3 and score > best_score:
                best_score = score
                best = r

        if best is None:
            debug["error"] = "헤더(1단/2단) 탐지 실패: 상단에 안전재고/현재고/품명/규격 등의 헤더가 보이지 않습니다."
            empty = pd.DataFrame(
                columns=["대분류", "중분류", "소분류", "품명", "규격", "안전재고", "현재고"]
            )
            return (empty, debug) if return_debug else empty

        df_tmp = raw.iloc[best + 1 :].copy()
        cols = [_clean_header_token(c) for c in raw.iloc[best].tolist()]
        # 빈 컬럼명 보정
        cols = [c if c != "" else f"col_{i}" for i, c in enumerate(cols)]
        df_tmp.columns = cols
        meta_tmp = {c: {"upper": c, "lower": ""} for c in df_tmp.columns}

        safe_col, curr_col = _try_map(df_tmp, meta_tmp)
        if safe_col is None or curr_col is None or safe_col == curr_col:
            debug["error"] = "안전재고/현재고 컬럼 매핑 실패(1단 헤더 fallback)."
            empty = pd.DataFrame(
                columns=["대분류", "중분류", "소분류", "품명", "규격", "안전재고", "현재고"]
            )
            return (empty, debug) if return_debug else empty

        chosen_df, chosen_meta = df_tmp, meta_tmp
        chosen_u, chosen_l = best, None
        chosen_upper, chosen_lower = cols, ["" for _ in cols]
        chosen_safe, chosen_curr = safe_col, curr_col

    # 4) 표준 컬럼 매핑
    assert chosen_df is not None
    assert chosen_meta is not None

    debug["chosen_upper_idx"] = chosen_u
    debug["chosen_lower_idx"] = chosen_l
    debug["upper_ffill_preview"] = (chosen_upper or [])[:15]
    debug["lower_preview"] = (chosen_lower or [])[:15]
    debug["combined_columns"] = list(chosen_df.columns)

    # 텍스트 컬럼 매핑
    col_major = _pick_text_column(chosen_df, ["대분류", "대 분류", "대분류명"])
    col_mid = _pick_text_column(chosen_df, ["중분류", "중 분류", "중분류명"])
    col_minor = _pick_text_column(chosen_df, ["소분류", "소 분류", "소분류명"])
    col_name = _pick_text_column(chosen_df, ["품명", "품목", "품목명", "자재명"])
    col_spec = _pick_text_column(chosen_df, ["규격", "사양", "SPEC"])

    safe_col = chosen_safe
    curr_col = chosen_curr

    if safe_col == curr_col:
        debug["error"] = "매핑 충돌: 안전재고와 현재고가 같은 열로 선택되었습니다."
        empty = pd.DataFrame(
            columns=["대분류", "중분류", "소분류", "품명", "규격", "안전재고", "현재고"]
        )
        return (empty, debug) if return_debug else empty

    debug["mapped_columns"] = {
        "대분류": col_major,
        "중분류": col_mid,
        "소분류": col_minor,
        "품명": col_name,
        "규격": col_spec,
        "안전재고": safe_col,
        "현재고": curr_col,
    }

    # 표준 DF 생성
    out = pd.DataFrame()
    out["대분류"] = chosen_df[col_major] if col_major in chosen_df.columns else ""
    out["중분류"] = chosen_df[col_mid] if col_mid in chosen_df.columns else ""
    out["소분류"] = chosen_df[col_minor] if col_minor in chosen_df.columns else ""
    out["품명"] = chosen_df[col_name] if col_name in chosen_df.columns else ""
    out["규격"] = chosen_df[col_spec] if col_spec in chosen_df.columns else ""

    # 숫자 컬럼
    def _to_num(series: pd.Series) -> pd.Series:
        s = series.copy()
        s = s.astype(str)
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace("\xa0", " ", regex=False)
        s = s.str.strip()
        s = s.replace({"": None, "nan": None, "None": None})
        return pd.to_numeric(s, errors="coerce")

    out["안전재고"] = _to_num(chosen_df[safe_col])
    out["현재고"] = _to_num(chosen_df[curr_col])

    # 행 정리: 품명/규격 둘 다 비어 있고, 안전/현재 모두 NaN이면 제거
    name_blank = out["품명"].astype(str).str.strip().replace("nan", "") == ""
    spec_blank = out["규격"].astype(str).str.strip().replace("nan", "") == ""
    num_blank = out["안전재고"].isna() & out["현재고"].isna()
    out = out[~(name_blank & spec_blank & num_blank)].copy()

    # 문자열 정리
    for c in ["대분류", "중분류", "소분류", "품명", "규격"]:
        out[c] = out[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        out.loc[out[c].str.lower() == "nan", c] = ""

    # 결측률 디버그
    try:
        debug["safe_missing_rate"] = float(out["안전재고"].isna().mean())
        debug["curr_missing_rate"] = float(out["현재고"].isna().mean())
        debug["safe_minmax"] = (float(out["안전재고"].min(skipna=True)), float(out["안전재고"].max(skipna=True)))
        debug["curr_minmax"] = (float(out["현재고"].min(skipna=True)), float(out["현재고"].max(skipna=True)))
    except Exception:
        pass

    return (out, debug) if return_debug else out
