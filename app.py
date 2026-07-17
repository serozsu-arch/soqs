"""SOQS v2.2 — Tek dosyalık bulut sürümü (Streamlit).

Tüm motor (metrics + scoring + engine + TEFAS updater) bu dosyada birleştirildi;
telefondan GitHub'a klasörsüz yükleme için hazırlanmıştır.
Çalıştırma: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================ METRICS ============================
TRADING_DAYS = 252


@dataclass(frozen=True)
class WindowMetrics:
    years: int
    observations: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    cagr: float
    volatility: float
    max_drawdown: float
    recovery_days: Optional[int]
    sharpe: float
    benchmark_cagr: Optional[float]
    alpha: Optional[float]


def _validate_series(series: pd.Series, name: str) -> pd.Series:
    s = pd.Series(series, copy=True).dropna().astype(float)
    if len(s) < 2:
        raise ValueError(f"{name} needs at least 2 valid observations")
    if (s <= 0).any():
        raise ValueError(f"{name} must contain only positive values")
    return s.sort_index()


def cagr(nav: pd.Series) -> float:
    s = _validate_series(nav, "NAV")
    elapsed_years = (s.index[-1] - s.index[0]).days / 365.25
    if elapsed_years <= 0:
        raise ValueError("NAV dates must span a positive period")
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / elapsed_years) - 1)


def annualized_volatility(nav: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    s = _validate_series(nav, "NAV")
    returns = s.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(trading_days))


def max_drawdown(nav: pd.Series) -> float:
    s = _validate_series(nav, "NAV")
    drawdown = s / s.cummax() - 1.0
    return float(drawdown.min())


def recovery_time_days(nav: pd.Series) -> Optional[int]:
    """Calendar days from pre-drawdown peak to first full recovery.

    Returns None if the maximum drawdown had not recovered by the last date.
    """
    s = _validate_series(nav, "NAV")
    running_max = s.cummax()
    dd = s / running_max - 1.0
    trough_date = dd.idxmin()
    peak_value = running_max.loc[trough_date]
    pre_trough = s.loc[:trough_date]
    peak_date = pre_trough[pre_trough == peak_value].index[-1]
    recovery_candidates = s.loc[trough_date:]
    recovery_candidates = recovery_candidates[recovery_candidates >= peak_value]
    if recovery_candidates.empty:
        return None
    recovery_date = recovery_candidates.index[0]
    return int((recovery_date - peak_date).days)


def sharpe_ratio(nav: pd.Series, annual_risk_free_rate: float = 0.0, trading_days: int = TRADING_DAYS) -> float:
    s = _validate_series(nav, "NAV")
    returns = s.pct_change().dropna()
    if returns.empty:
        return 0.0
    daily_rf = (1 + annual_risk_free_rate) ** (1 / trading_days) - 1
    excess = returns - daily_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(trading_days))


def window_slice(series: pd.Series, years: int, end_date: Optional[pd.Timestamp] = None) -> pd.Series:
    s = _validate_series(series, "series")
    end = pd.Timestamp(end_date) if end_date is not None else s.index.max()
    start = end - pd.DateOffset(years=years)
    sliced = s.loc[(s.index >= start) & (s.index <= end)]
    if len(sliced) < 2:
        raise ValueError(f"Not enough data for {years}Y window")
    return sliced


def calculate_window_metrics(
    nav: pd.Series,
    years: int,
    benchmark_nav: Optional[pd.Series] = None,
    annual_risk_free_rate: float = 0.0,
) -> WindowMetrics:
    nav_w = window_slice(nav, years)
    bench_cagr = None
    alpha = None
    if benchmark_nav is not None:
        aligned = pd.concat(
            [nav.rename("fund"), benchmark_nav.rename("benchmark")], axis=1
        ).dropna()
        if len(aligned) >= 2:
            end = aligned.index.max()
            start = end - pd.DateOffset(years=years)
            aligned = aligned.loc[(aligned.index >= start) & (aligned.index <= end)]
            if len(aligned) >= 2:
                nav_w = aligned["fund"]
                bench_cagr = cagr(aligned["benchmark"])
                alpha = cagr(nav_w) - bench_cagr

    return WindowMetrics(
        years=years,
        observations=len(nav_w),
        start_date=nav_w.index.min(),
        end_date=nav_w.index.max(),
        cagr=cagr(nav_w),
        volatility=annualized_volatility(nav_w),
        max_drawdown=max_drawdown(nav_w),
        recovery_days=recovery_time_days(nav_w),
        sharpe=sharpe_ratio(nav_w, annual_risk_free_rate),
        benchmark_cagr=bench_cagr,
        alpha=alpha,
    )

# ============================ SCORING ============================
DEFAULT_WEIGHTS = {
    "performance": 0.40,
    "risk": 0.30,
    "risk_adjusted": 0.30,
}


def _percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(np.where(s.notna(), 50.0, np.nan), index=s.index)
    rank = s.rank(method="average", pct=True, ascending=True) * 100
    return rank if higher_is_better else 100 - rank + (100 / s.notna().sum())


def _mean_available(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return pd.Series(np.nan, index=frame.index)
    return frame[cols].mean(axis=1, skipna=True)


def grade_from_score(score: float) -> str:
    if pd.isna(score):
        return "Not Rated"
    if score >= 90:
        return "Elite"
    if score >= 80:
        return "Platinum"
    if score >= 70:
        return "Gold"
    if score >= 60:
        return "Silver"
    if score >= 50:
        return "Bronze"
    return "Watch"


def score_league(league_df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    required_total = sum(weights.values())
    if not np.isclose(required_total, 1.0):
        raise ValueError("SOQS weights must sum to 1.0")

    df = league_df.copy()

    perf_cols = [c for c in ["cagr_1y", "cagr_3y", "cagr_5y", "alpha_1y", "alpha_3y", "alpha_5y"] if c in df]
    risk_cols = [c for c in ["volatility_1y", "volatility_3y", "volatility_5y", "max_drawdown_1y", "max_drawdown_3y", "max_drawdown_5y", "recovery_days_1y", "recovery_days_3y", "recovery_days_5y"] if c in df]
    ra_cols = [c for c in ["sharpe_1y", "sharpe_3y", "sharpe_5y"] if c in df]

    perf_scores = pd.DataFrame(index=df.index)
    for c in perf_cols:
        perf_scores[c] = _percentile_score(df[c], True)

    risk_scores = pd.DataFrame(index=df.index)
    for c in risk_cols:
        higher = c.startswith("max_drawdown")  # less negative is better
        if c.startswith("recovery_days"):
            higher = False
        risk_scores[c] = _percentile_score(df[c], higher)

    ra_scores = pd.DataFrame(index=df.index)
    for c in ra_cols:
        ra_scores[c] = _percentile_score(df[c], True)

    df["performance_score"] = perf_scores.mean(axis=1, skipna=True)
    df["risk_score"] = risk_scores.mean(axis=1, skipna=True)
    df["risk_adjusted_score"] = ra_scores.mean(axis=1, skipna=True)

    components = ["performance_score", "risk_score", "risk_adjusted_score"]
    component_weights = pd.Series(
        [weights["performance"], weights["risk"], weights["risk_adjusted"]],
        index=components,
    )
    available = df[components].notna()
    weighted = df[components].mul(component_weights, axis=1)
    denominator = available.mul(component_weights, axis=1).sum(axis=1)
    df["soqs_score"] = weighted.sum(axis=1, skipna=True) / denominator.replace(0, np.nan)
    df["grade"] = df["soqs_score"].map(grade_from_score)
    df["league_rank"] = df["soqs_score"].rank(method="min", ascending=False).astype("Int64")
    return df.sort_values(["league_rank", "code"])


def score_all_leagues(metrics_df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    if "league" not in metrics_df.columns:
        raise ValueError("metrics dataframe must contain a 'league' column")
    parts = [score_league(group, weights) for _, group in metrics_df.groupby("league", sort=True)]
    return pd.concat(parts, ignore_index=True) if parts else metrics_df.copy()

# ============================ ENGINE ============================
def load_inputs(fund_master_path: str | Path, prices_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(fund_master_path)
    prices = pd.read_csv(prices_path, parse_dates=["date"])

    master_required = {"code", "name", "league", "benchmark"}
    prices_required = {"date", "code", "nav"}
    missing_master = master_required - set(master.columns)
    missing_prices = prices_required - set(prices.columns)
    if missing_master:
        raise ValueError(f"fund_master missing columns: {sorted(missing_master)}")
    if missing_prices:
        raise ValueError(f"prices missing columns: {sorted(missing_prices)}")

    if master["code"].duplicated().any():
        duplicates = master.loc[master["code"].duplicated(), "code"].tolist()
        raise ValueError(f"Duplicate fund codes in fund_master: {duplicates}")
    if prices[["date", "code"]].duplicated().any():
        raise ValueError("Duplicate date/code rows detected in prices")
    if (pd.to_numeric(prices["nav"], errors="coerce") <= 0).any():
        raise ValueError("NAV must be positive")

    prices["nav"] = pd.to_numeric(prices["nav"], errors="raise")
    if "benchmark_nav" in prices.columns:
        prices["benchmark_nav"] = pd.to_numeric(prices["benchmark_nav"], errors="coerce")
    return master, prices.sort_values(["code", "date"])


def calculate_fund_metrics(
    master: pd.DataFrame,
    prices: pd.DataFrame,
    windows: Iterable[int] = (1, 3, 5),
    annual_risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    records: list[dict] = []
    for _, fund in master.iterrows():
        code = fund["code"]
        fund_prices = prices.loc[prices["code"] == code].set_index("date").sort_index()
        record = fund.to_dict()
        record["data_start"] = fund_prices.index.min() if not fund_prices.empty else pd.NaT
        record["data_end"] = fund_prices.index.max() if not fund_prices.empty else pd.NaT
        record["observations"] = len(fund_prices)

        if fund_prices.empty:
            record["status"] = "NO_DATA"
            records.append(record)
            continue

        benchmark = fund_prices["benchmark_nav"] if "benchmark_nav" in fund_prices else None
        for years in windows:
            suffix = f"{years}y"
            try:
                m = calculate_window_metrics(
                    fund_prices["nav"],
                    years=years,
                    benchmark_nav=benchmark,
                    annual_risk_free_rate=annual_risk_free_rate,
                )
                record.update(
                    {
                        f"cagr_{suffix}": m.cagr,
                        f"volatility_{suffix}": m.volatility,
                        f"max_drawdown_{suffix}": m.max_drawdown,
                        f"recovery_days_{suffix}": m.recovery_days,
                        f"sharpe_{suffix}": m.sharpe,
                        f"benchmark_cagr_{suffix}": m.benchmark_cagr,
                        f"alpha_{suffix}": m.alpha,
                        f"observations_{suffix}": m.observations,
                    }
                )
            except ValueError:
                for metric in ["cagr", "volatility", "max_drawdown", "recovery_days", "sharpe", "benchmark_cagr", "alpha", "observations"]:
                    record[f"{metric}_{suffix}"] = pd.NA
        record["status"] = "OK"
        records.append(record)

    return pd.DataFrame(records)


def build_hall_of_fame(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    leaders = scored.sort_values(["league", "league_rank"]).groupby("league", as_index=False).first()
    return leaders[["league", "code", "name", "soqs_score", "grade"]].rename(columns={"code": "champion_code", "name": "champion_name"})


def export_reports(scored: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hof = build_hall_of_fame(scored)

    csv_path = out / "soqs_rankings.csv"
    xlsx_path = out / "soqs_report.xlsx"
    hof_path = out / "hall_of_fame.csv"

    scored.to_csv(csv_path, index=False)
    hof.to_csv(hof_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        scored.to_excel(writer, sheet_name="Rankings", index=False)
        hof.to_excel(writer, sheet_name="Hall of Fame", index=False)
        for league, group in scored.groupby("league"):
            safe_name = str(league)[:31]
            group.to_excel(writer, sheet_name=safe_name, index=False)

    return {"rankings_csv": csv_path, "hall_of_fame_csv": hof_path, "excel": xlsx_path}


def run_engine(
    fund_master_path: str | Path,
    prices_path: str | Path,
    output_dir: str | Path,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    master, prices = load_inputs(fund_master_path, prices_path)
    metrics = calculate_fund_metrics(master, prices, annual_risk_free_rate=annual_risk_free_rate)
    scored = score_all_leagues(metrics)
    paths = export_reports(scored, output_dir)
    return scored, paths

# ============================ TEFAS UPDATER ============================
TEFAS_HISTORY_URL = "https://www.tefas.gov.tr/api/DB/BindHistoryInfo"


@dataclass(frozen=True)
class DownloadResult:
    prices: pd.DataFrame
    requested_codes: tuple[str, ...]
    missing_codes: tuple[str, ...]


def _date_text(value: date | datetime | str) -> str:
    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="raise")
        return parsed.strftime("%d.%m.%Y")
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%d.%m.%Y")


def _extract_rows(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("data", "Data", "rows", "Rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # Some responses wrap JSON as a string.
        for key in ("d", "result", "Result"):
            value = payload.get(key)
            if isinstance(value, str):
                try:
                    return _extract_rows(json.loads(value))
                except json.JSONDecodeError:
                    pass
            if isinstance(value, (dict, list)):
                return _extract_rows(value)
    if isinstance(payload, list):
        return payload
    return []


def _normalize_tefas_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "code", "nav"])

    raw = pd.DataFrame(rows)
    aliases = {
        "date": ("TARIH", "Tarih", "tarih", "DATE", "Date", "date"),
        "code": ("FONKODU", "FonKodu", "fonkodu", "FON_KODU", "code"),
        "nav": ("FIYAT", "Fiyat", "fiyat", "PRICE", "Price", "nav"),
    }
    selected: dict[str, str] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in raw.columns:
                selected[target] = candidate
                break
        if target not in selected:
            raise ValueError(
                f"TEFAS response does not contain a recognizable {target} column. "
                f"Received columns: {list(raw.columns)}"
            )

    out = raw[[selected["date"], selected["code"], selected["nav"]]].copy()
    out.columns = ["date", "code", "nav"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["code"] = out["code"].astype(str).str.strip().str.upper()
    def parse_number(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(" ", "")
        if "," in text and "." in text:
            # Turkish thousands separator + decimal comma, e.g. 1.234,56
            text = text.replace(".", "").replace(",", ".")
        elif "," in text:
            text = text.replace(",", ".")
        try:
            return float(text)
        except ValueError:
            return None

    out["nav"] = out["nav"].map(parse_number)
    out = out.dropna(subset=["date", "code", "nav"])
    out = out.loc[out["nav"] > 0]
    return out.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])


def fetch_tefas_history(
    codes: Iterable[str],
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    fund_type: str = "YAT",
    *,
    session: requests.Session | None = None,
    timeout: int = 45,
    attempts: int = 3,
    pause_seconds: float = 1.0,
) -> DownloadResult:
    """Download selected fund histories from TEFAS.

    fund_type commonly uses YAT for investment funds and EMK for pension funds.
    The endpoint is public but undocumented, so failures are surfaced clearly.
    """
    normalized_codes = tuple(dict.fromkeys(str(c).strip().upper() for c in codes if str(c).strip()))
    if not normalized_codes:
        raise ValueError("At least one fund code is required")

    own_session = session is None
    client = session or requests.Session()
    client.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Origin": "https://www.tefas.gov.tr",
            "Referer": "https://www.tefas.gov.tr/TarihselVeriler.aspx",
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    # WAF çerezlerini almak için önce sayfayı normal bir tarayıcı gibi ziyaret et
    for warmup_url in (
        "https://www.tefas.gov.tr/TarihselVeriler.aspx",
        "https://www.tefas.gov.tr/BesTarihselVeriler.aspx",
    ):
        try:
            client.get(warmup_url, timeout=timeout)
        except requests.RequestException:
            pass

    frames: list[pd.DataFrame] = []
    failed_batches: list[tuple[tuple[str, ...], str]] = []
    try:
        # TEFAS accepts a comma-separated code list. Smaller batches are more reliable.
        batch_size = 25
        for offset in range(0, len(normalized_codes), batch_size):
            batch = normalized_codes[offset : offset + batch_size]
            form = {
                "fontip": fund_type,
                "bastarih": _date_text(start_date),
                "bittarih": _date_text(end_date),
                "fonkod": ",".join(batch),
            }
            last_error: Exception | None = None
            referers = (
                "https://www.tefas.gov.tr/TarihselVeriler.aspx",
                "https://www.tefas.gov.tr/BesTarihselVeriler.aspx",
            )
            if fund_type.upper() == "EMK":
                referers = tuple(reversed(referers))
            for attempt in range(1, attempts + 1):
                try:
                    headers = {"Referer": referers[(attempt - 1) % len(referers)]}
                    response = client.post(TEFAS_HISTORY_URL, data=form, headers=headers, timeout=timeout)
                    response.raise_for_status()
                    rows = _extract_rows(response.json())
                    frame = _normalize_tefas_rows(rows)
                    if not frame.empty:
                        frame = frame.loc[frame["code"].isin(batch)]
                    frames.append(frame)
                    last_error = None
                    break
                except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    if attempt < attempts:
                        time.sleep(pause_seconds * attempt)
            if last_error is not None:
                # Bu grup indirilemedi; uygulamayı düşürme, eksikler listesinde raporlanacak.
                failed_batches.append((tuple(batch), str(last_error)))
            if offset + batch_size < len(normalized_codes):
                time.sleep(pause_seconds)
    finally:
        if own_session:
            client.close()

    prices = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["date", "code", "nav"])
    )
    prices = prices.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])
    downloaded = set(prices["code"].unique()) if not prices.empty else set()
    missing = tuple(code for code in normalized_codes if code not in downloaded)
    return DownloadResult(prices=prices, requested_codes=normalized_codes, missing_codes=missing)


def merge_prices(existing_path: str | Path, new_prices: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    existing_path = Path(existing_path)
    destination = Path(output_path) if output_path else existing_path

    if existing_path.exists() and existing_path.stat().st_size:
        existing = pd.read_csv(existing_path, parse_dates=["date"])
    else:
        existing = pd.DataFrame(columns=["date", "code", "nav"])

    required = {"date", "code", "nav"}
    if not required.issubset(new_prices.columns):
        raise ValueError(f"new_prices must include {sorted(required)}")

    combined = pd.concat([existing, new_prices], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    combined["code"] = combined["code"].astype(str).str.strip().str.upper()
    combined["nav"] = pd.to_numeric(combined["nav"], errors="raise")
    combined = combined.drop_duplicates(["date", "code"], keep="last")
    combined = combined.sort_values(["code", "date"]).reset_index(drop=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    export = combined.copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    export.to_csv(destination, index=False)
    return combined


def default_start_date(years: int = 6) -> date:
    return date.today() - timedelta(days=366 * years)

# ============================ STREAMLIT UI ============================
BASE = Path(__file__).parent
FUND_MASTER = BASE / "fund_master.csv"
PRICES = BASE / "prices.csv"
OUTPUT = BASE / "output"

GRADE_COLORS = {
    "Elite": "#7C3AED",
    "Platinum": "#0EA5E9",
    "Gold": "#D4A017",
    "Silver": "#94A3B8",
    "Bronze": "#B45309",
    "Watch": "#DC2626",
    "Not Rated": "#64748B",
}

st.set_page_config(page_title="SOQS Fon Ligi", page_icon="🏆", layout="wide")


# --- Şifre koruması (yalnızca Secrets'ta APP_PASSWORD tanımlıysa devreye girer) ---
def _check_password() -> bool:
    try:
        expected = st.secrets.get("APP_PASSWORD", "")
    except Exception:  # yerelde secrets.toml yoksa
        expected = ""
    if not expected:
        return True  # şifre tanımlı değilse (yerel kullanım) serbest geç
    if st.session_state.get("auth_ok"):
        return True
    st.title("🏆 SOQS Fon Ligi")
    pwd = st.text_input("Şifre", type="password")
    if pwd:
        if pwd == expected:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Şifre yanlış.")
    return False


if not _check_password():
    st.stop()

st.markdown(
    """
    <style>
    .grade-badge {display:inline-block;padding:2px 10px;border-radius:12px;
                  color:white;font-weight:600;font-size:0.85rem;}
    .hof-card {border:1px solid #334155;border-radius:10px;padding:14px 16px;
               background:linear-gradient(160deg,#1E293B,#0F172A);margin-bottom:8px;}
    .hof-league {font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;color:#94A3B8;}
    .hof-name {font-size:1.0rem;font-weight:700;color:#F1F5F9;margin:2px 0;}
    .hof-code {font-family:monospace;color:#38BDF8;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- yardımcılar -----------------------------

def load_master() -> pd.DataFrame:
    return pd.read_csv(FUND_MASTER)


def badge(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#64748B")
    return f'<span class="grade-badge" style="background:{color}">{grade}</span>'


@st.cache_data(show_spinner=False)
def load_prices(mtime: float) -> pd.DataFrame:
    return pd.read_csv(PRICES, parse_dates=["date"])


def prices_mtime() -> float:
    return PRICES.stat().st_mtime if PRICES.exists() else 0.0


def update_prices(start: date, end: date, codes_filter: list[str] | None, status) -> None:
    master = load_master()
    if codes_filter:
        master = master[master["code"].isin(codes_filter)]
    groups = master.groupby(master.get("source_type", pd.Series("YAT", index=master.index)).fillna("YAT"))
    all_frames = []
    missing: list[str] = []
    for fund_type, group in groups:
        codes = group["code"].tolist()
        status.write(f"⬇️ TEFAS'tan indiriliyor: {fund_type} ({len(codes)} fon)…")
        result = fetch_tefas_history(codes, start, end, fund_type=str(fund_type))
        all_frames.append(result.prices)
        missing.extend(result.missing_codes)
    new_prices = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if new_prices.empty:
        status.error("TEFAS'tan hiç veri gelmedi. Fon kodlarını ve interneti kontrol edin.")
        return
    merge_prices(PRICES, new_prices)
    load_prices.clear()
    ok_msg = f"✅ {new_prices['code'].nunique()} fon güncellendi ({len(new_prices):,} satır)."
    if missing:
        ok_msg += f" ⚠️ Veri gelmeyen kodlar: {', '.join(missing)}"
    status.success(ok_msg)
    if missing:
        status.warning(
            "Eksik kodlar için TEFAS isteği başarısız olmuş olabilir (ör. BES fonlarında geçici 404). "
            "İnen verilerle skorlamaya devam edebilirsiniz; eksikler için daha sonra tekrar deneyin."
        )


def run_scoring(risk_free: float) -> tuple[pd.DataFrame, dict]:
    scored, paths = run_engine(FUND_MASTER, PRICES, OUTPUT, annual_risk_free_rate=risk_free)
    return scored, paths


# ----------------------------- kenar çubuğu -----------------------------

with st.sidebar:
    st.title("🏆 SOQS v2.1")
    st.caption("Fon Skorlama ve Lig Sistemi")

    master_df = load_master() if FUND_MASTER.exists() else pd.DataFrame()
    st.metric("Ana listedeki fon", len(master_df))

    st.divider()
    st.subheader("1 · Veri Güncelle (TEFAS)")
    col1, col2 = st.columns(2)
    start_d = col1.date_input("Başlangıç", value=default_start_date())
    end_d = col2.date_input("Bitiş", value=date.today())
    code_options = master_df["code"].tolist() if not master_df.empty else []
    sel_codes = st.multiselect("Yalnızca bu fonlar (boş = hepsi)", code_options)
    if st.button("⬇️ TEFAS'tan Güncelle", use_container_width=True):
        with st.status("Veri indiriliyor…", expanded=True) as status:
            try:
                update_prices(start_d, end_d, sel_codes or None, status)
            except Exception as exc:  # noqa: BLE001
                status.error(f"Güncelleme hatası: {exc}")

    st.divider()
    st.subheader("2 · Skorlama")
    risk_free = st.number_input(
        "Yıllık risksiz faiz (ör. 0.40 = %40)", min_value=0.0, max_value=2.0, value=0.0, step=0.05
    )
    run_clicked = st.button("⚙️ SOQS Motorunu Çalıştır", type="primary", use_container_width=True)

# ----------------------------- ana ekran -----------------------------

if not FUND_MASTER.exists():
    st.error("data/fund_master.csv bulunamadı. Önce fon listenizi hazırlayın.")
    st.stop()

if not PRICES.exists():
    st.warning("Henüz fiyat verisi yok. Soldan **TEFAS'tan Güncelle** ile başlayın.")
    st.stop()

if run_clicked or "scored" not in st.session_state:
    with st.spinner("Metrikler hesaplanıyor, ligler skorlanıyor…"):
        try:
            scored, paths = run_scoring(risk_free)
            st.session_state["scored"] = scored
            st.session_state["paths"] = paths
        except Exception as exc:  # noqa: BLE001
            st.error(f"Motor hatası: {exc}")
            st.stop()

scored: pd.DataFrame = st.session_state["scored"]
prices_df = load_prices(prices_mtime())

# --- Veri durumu şeridi ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fon", scored["code"].nunique())
c2.metric("Lig", scored["league"].nunique())
c3.metric("Son fiyat tarihi", str(prices_df["date"].max().date()))
c4.metric("Fiyat satırı", f"{len(prices_df):,}")

# --- Hall of Fame ---
st.subheader("🥇 Lig Şampiyonları")
hof = build_hall_of_fame(scored)
cols = st.columns(min(4, max(1, len(hof))))
for i, (_, row) in enumerate(hof.iterrows()):
    with cols[i % len(cols)]:
        st.markdown(
            f"""<div class="hof-card">
            <div class="hof-league">{row['league']}</div>
            <div class="hof-name">{row['champion_name']}</div>
            <span class="hof-code">{row['champion_code']}</span> ·
            <b>{row['soqs_score']:.1f}</b> {badge(row['grade'])}
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# --- Lig sekmesi ---
leagues = ["Tüm Ligler"] + sorted(scored["league"].unique().tolist())
sel_league = st.selectbox("Lig seçin", leagues)
view = scored if sel_league == "Tüm Ligler" else scored[scored["league"] == sel_league]

tab_table, tab_charts, tab_nav = st.tabs(["📋 Sıralama", "📊 Metrikler", "📈 Fiyat Grafiği"])

with tab_table:
    show_cols = [
        "league", "league_rank", "code", "name", "soqs_score", "grade",
        "performance_score", "risk_score", "risk_adjusted_score",
        "cagr_1y", "cagr_3y", "cagr_5y", "sharpe_1y", "sharpe_3y",
        "volatility_1y", "max_drawdown_1y", "observations",
    ]
    show_cols = [c for c in show_cols if c in view.columns]
    table = view[show_cols].copy()
    pct_cols = [c for c in table.columns if c.startswith(("cagr", "volatility", "max_drawdown"))]
    for c in pct_cols:
        table[c] = (pd.to_numeric(table[c], errors="coerce") * 100).round(2)
    st.dataframe(
        table.style.background_gradient(subset=["soqs_score"], cmap="RdYlGn", vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True,
    )
    xlsx_path = OUTPUT / "soqs_report.xlsx"
    if xlsx_path.exists():
        st.download_button(
            "⬇️ Excel Raporunu İndir (soqs_report.xlsx)",
            data=xlsx_path.read_bytes(),
            file_name="soqs_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with tab_charts:
    metric_map = {
        "SOQS Skoru": "soqs_score",
        "CAGR 1Y (%)": "cagr_1y",
        "CAGR 3Y (%)": "cagr_3y",
        "Sharpe 1Y": "sharpe_1y",
        "Sharpe 3Y": "sharpe_3y",
        "Volatilite 1Y (%)": "volatility_1y",
        "Maks. Düşüş 1Y (%)": "max_drawdown_1y",
    }
    metric_label = st.selectbox("Metrik", list(metric_map.keys()))
    mcol = metric_map[metric_label]
    plot_df = view.dropna(subset=[mcol]).copy()
    if mcol.startswith(("cagr", "volatility", "max_drawdown")):
        plot_df[mcol] = plot_df[mcol] * 100
    fig = px.bar(
        plot_df.sort_values(mcol, ascending=False),
        x="code", y=mcol, color="grade",
        color_discrete_map=GRADE_COLORS,
        hover_data=["name", "league"],
        labels={mcol: metric_label, "code": "Fon"},
    )
    fig.update_layout(height=420, legend_title="Derece")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Risk – Getiri Haritası (3Y)**")
    if {"volatility_3y", "cagr_3y"}.issubset(view.columns):
        sc = view.dropna(subset=["volatility_3y", "cagr_3y"]).copy()
        sc["cagr_3y"] *= 100
        sc["volatility_3y"] *= 100
        fig2 = px.scatter(
            sc, x="volatility_3y", y="cagr_3y", color="grade", text="code",
            color_discrete_map=GRADE_COLORS, hover_data=["name", "soqs_score"],
            labels={"volatility_3y": "Volatilite 3Y (%)", "cagr_3y": "CAGR 3Y (%)"},
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=460, legend_title="Derece")
        st.plotly_chart(fig2, use_container_width=True)

with tab_nav:
    nav_codes = st.multiselect(
        "Fon seçin (normalize edilmiş fiyat, başlangıç = 100)",
        view["code"].tolist(),
        default=view["code"].tolist()[:5],
    )
    if nav_codes:
        fig3 = go.Figure()
        for code in nav_codes:
            s = prices_df[prices_df["code"] == code].sort_values("date")
            if s.empty:
                continue
            norm = s["nav"] / s["nav"].iloc[0] * 100
            fig3.add_trace(go.Scatter(x=s["date"], y=norm, mode="lines", name=code))
        fig3.update_layout(height=460, yaxis_title="Endeks (başlangıç=100)", xaxis_title="Tarih")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Grafik için en az bir fon seçin.")
