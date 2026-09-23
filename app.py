"""
Saudi stock scenarios: pick companies, see crash / likely / good cases over the next 5 years.

Run locally:  streamlit run app.py
Needs data/prices.parquet, data/companies.csv (build_dataset.py) and
data/scenario_calibration.json (calibrate_scenarios.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.montecarlo import MonteCarlo, scale_curve

DATA = Path("data")
MAX_YEARS = 5
DAYS_PER_MONTH = 30.44
LTR = "‎"                    # keeps "-75%" from being flipped to "75%-" in Arabic text
COLORS = {"crash": "#d64545", "likely": "#2f6fdb", "good": "#2e9d5b", "history": "#555555",
          "band": "rgba(47,111,219,0.12)"}

T = {
    "en": {
        "title": "Saudi Stock Scenarios",
        "subtitle": "Where could a Tadawul stock be over the next 5 years? Three scenarios, tested on 20 years of history.",
        "pick": "Companies", "pick_help": "Choose one or more companies",
        "lang": "العربية",
        "crash": "Crash case", "likely": "Likely", "good": "Good case", "history": "Past price",
        "today": "Today", "company": "Company", "price_today": "Today (SAR)",
        "in_years": "In 5 years",
        "summary": "Summary", "no_pick": "Choose at least one company to see its scenarios.",
        "what_title": "How to read this",
        "what": (
            "- **Likely** is today's price. In our tests, no method predicted the direction of Saudi "
            "stocks better than assuming they stay where they are.\n"
            "- **Crash case** is sized to hold even through the 2006 market crash: in every year we "
            "tested since 2005, at most 1 in 10 stocks ended below it.\n"
            "- **Good case**: about 1 in 10 stocks ended above it. Beating it is possible, especially "
            "in a boom.\n"
            "- The shaded area is where the price stayed about 9 times out of 10 in our tests."
        ),
        "warning": (
            "**This is not a prediction or financial advice.** Nobody can reliably predict stock "
            "prices. When the whole market crashes, almost every stock falls together, and in 2006 "
            "many Saudi stocks lost more than half their value within a year. The crash case is "
            "built to cover that, but a worse crash is always possible."
        ),
        "tested": "Tested on {n:,} past forecasts made {origins}: {below:.0%} ended below the crash case, "
                  "{above:.0%} above the good case.",
        "short": "{name} has only {years:.1f} years of history, so its scenarios are less reliable.",
        "no_2006": "{name} was listed after the 2006 crash, so its own history has no crash like it. "
                   "Its crash case is widened using what happened to other stocks in 2006.",
        "data_date": "Prices up to {date}.",
    },
    "ar": {
        "title": "سيناريوهات الأسهم السعودية",
        "subtitle": "أين قد يكون سعر السهم في تداول خلال السنوات الخمس القادمة؟ ثلاثة سيناريوهات مختبرة على 20 سنة من البيانات.",
        "pick": "الشركات", "pick_help": "اختر شركة أو أكثر",
        "lang": "English",
        "crash": "حالة الانهيار", "likely": "المتوقع", "good": "الحالة الجيدة", "history": "السعر السابق",
        "today": "اليوم", "company": "الشركة", "price_today": "اليوم (ريال)",
        "in_years": "بعد 5 سنوات",
        "summary": "ملخص", "no_pick": "اختر شركة واحدة على الأقل لعرض السيناريوهات.",
        "what_title": "كيف تقرأ هذه الأرقام",
        "what": (
            "- **المتوقع** هو سعر اليوم. في اختباراتنا لم تتفوق أي طريقة في توقع اتجاه الأسهم السعودية "
            "على افتراض بقاء السعر كما هو.\n"
            "- **حالة الانهيار** مصممة لتصمد حتى في انهيار السوق عام 2006: في كل سنة اختبرناها منذ 2005، "
            "انتهى سهم واحد من كل 10 أسهم على الأكثر تحتها.\n"
            "- **الحالة الجيدة**: انتهى سهم واحد تقريباً من كل 10 أسهم فوقها، وتجاوزها ممكن خاصة في فترات الارتفاع.\n"
            "- المنطقة المظللة هي المكان الذي بقي فيه السعر تقريباً 9 مرات من كل 10 في اختباراتنا."
        ),
        "warning": (
            "**هذه ليست توقعات مؤكدة ولا نصيحة مالية.** لا أحد يستطيع توقع أسعار الأسهم بدقة. "
            "عندما ينهار السوق كله تنخفض معظم الأسهم معاً، وفي عام 2006 خسرت أسهم سعودية كثيرة أكثر "
            "من نصف قيمتها خلال سنة. حالة الانهيار مصممة لتغطية ذلك، لكن انهياراً أسوأ ممكن دائماً."
        ),
        "tested": "تم اختبارها على {n:,} توقع سابق بين {origins}: {below:.0%} انتهت تحت حالة الانهيار، "
                  "و{above:.0%} فوق الحالة الجيدة.",
        "short": "لدى {name} بيانات لمدة {years:.1f} سنوات فقط، لذلك سيناريوهاتها أقل موثوقية.",
        "no_2006": "أُدرجت {name} بعد انهيار 2006، لذلك لا يحتوي تاريخها على انهيار مماثل. "
                   "تم توسيع حالة الانهيار لها بناءً على ما حدث للأسهم الأخرى في 2006.",
        "data_date": "الأسعار حتى {date}.",
    },
}


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_companies() -> pd.DataFrame:
    c = pd.read_csv(DATA / "companies.csv", parse_dates=["first_date", "last_date"])
    c = c[c["active"]].sort_values("name")
    c["label"] = c["name"] + " (" + c["ticker"].str.replace(".SR", "", regex=False) + ")"
    return c


@st.cache_data
def load_calibration() -> dict:
    return json.loads((DATA / "scenario_calibration.json").read_text())


@st.cache_data
def load_close(ticker: str) -> pd.Series:
    p = pd.read_parquet(DATA / "prices.parquet", columns=["date", "ticker", "close"],
                        filters=[("ticker", "==", ticker)])
    return p.set_index("date")["close"].sort_index()


@st.cache_data
def scenarios(ticker: str, last_date: pd.Timestamp) -> pd.DataFrame:
    """Monthly crash/likely/good prices for 5 years. last_date is in the key so new data refreshes it."""
    cal = load_calibration()
    months = MAX_YEARS * 12
    mc = MonteCarlo(cal["drift"], scale_curve(cal, "lo", months), scale_curve(cal, "hi", months),
                    n_paths=5000, seed=0).fit(load_close(ticker))
    s = mc.scenarios(months)
    s["date"] = last_date + pd.to_timedelta(s["month"] * DAYS_PER_MONTH, unit="D")
    return s.rename(columns={"worst": "crash", "best": "good"})


def pct(x: float) -> str:
    """Signed percent for the table: '+36%', '-52%', and '0%' rather than '-0%'."""
    return "0%" if round(x, 2) == 0 else LTR + f"{x:+.0%}"


def tested_note(cal: dict, t: dict) -> str:
    """Backtest result at the chart's end (5 years)."""
    h = f"{MAX_YEARS}y"
    r = cal["tested"][h]
    return t["tested"].format(n=r["n"], origins=r["origins"].replace("-", "–"),
                              below=r["below_worst"], above=r["above_best"]) + f" ({h})"


# ----------------------------------------------------------------------------- chart
def chart(name: str, close: pd.Series, sc: pd.DataFrame, t: dict) -> go.Figure:
    hist = close[close.index >= close.index[-1] - pd.DateOffset(years=2)]
    today = pd.DataFrame({"date": [close.index[-1]], "crash": [close.iloc[-1]],
                          "likely": [close.iloc[-1]], "good": [close.iloc[-1]]})
    sc = pd.concat([today, sc], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name=t["history"],
                             line=dict(color=COLORS["history"], width=1.5),
                             hovertemplate="%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=sc["date"], y=sc["good"], name=t["good"],
                             line=dict(color=COLORS["good"], width=2),
                             hovertemplate="%{y:.2f}<extra>" + t["good"] + "</extra>"))
    fig.add_trace(go.Scatter(x=sc["date"], y=sc["crash"], name=t["crash"], fill="tonexty",
                             fillcolor=COLORS["band"], line=dict(color=COLORS["crash"], width=2),
                             hovertemplate="%{y:.2f}<extra>" + t["crash"] + "</extra>"))
    fig.add_trace(go.Scatter(x=sc["date"], y=sc["likely"], name=t["likely"],
                             line=dict(color=COLORS["likely"], width=2, dash="dash"),
                             hovertemplate="%{y:.2f}<extra>" + t["likely"] + "</extra>"))
    fig.update_layout(title=dict(text=name, font=dict(size=16)), height=360,
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
                      yaxis_title="SAR", dragmode=False)
    return fig


# ----------------------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="Saudi Stock Scenarios", page_icon="📈", layout="centered")
    if "lang" not in st.session_state:
        st.session_state.lang = "ar"
    t = T[st.session_state.lang]
    if st.session_state.lang == "ar":
        st.markdown("<style>.stMainBlockContainer, [data-testid='stSidebar'] {direction: rtl; "
                    "text-align: right;} .js-plotly-plot {direction: ltr;}</style>",
                    unsafe_allow_html=True)

    top = st.columns([4, 1])
    top[0].title(t["title"])
    if top[1].button(t["lang"], use_container_width=True):
        st.session_state.lang = "en" if st.session_state.lang == "ar" else "ar"
        st.rerun()
    st.caption(t["subtitle"])

    companies = load_companies()
    cal = load_calibration()
    labels = companies.set_index("label")["ticker"]
    default = [lb for lb in labels.index if labels[lb] in ("2222.SR", "1120.SR")]
    picked = st.multiselect(t["pick"], labels.index, default=default, help=t["pick_help"])

    st.warning(t["warning"])
    if not picked:
        st.info(t["no_pick"])
        return

    rows, figs, notes = [], [], []
    for label in picked:
        info = companies[companies["label"] == label].iloc[0]
        close = load_close(info["ticker"])
        sc = scenarios(info["ticker"], close.index[-1])
        end, now = sc.iloc[-1], close.iloc[-1]
        # Percent changes keep the table narrow enough for a phone; prices are on the charts
        rows.append({t["company"]: label, t["price_today"]: f"{now:.2f}",
                     **{t[k]: pct(end[k] / now - 1) for k in ("crash", "likely", "good")}})
        figs.append(chart(label, close, sc, t))
        hist_years = (close.index[-1] - close.index[0]).days / 365.25
        if hist_years < 5:
            notes.append(t["short"].format(name=info["name"], years=hist_years))
        elif close.index[0] > pd.Timestamp("2006-01-01"):
            notes.append(t["no_2006"].format(name=info["name"]))

    st.subheader(t["summary"] + " · " + t["in_years"])
    st.table(pd.DataFrame(rows).set_index(t["company"]))
    for note in notes:
        st.caption("⚠️ " + note)

    for fig in figs:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander(t["what_title"], expanded=True):
        st.markdown(t["what"])
        st.caption(tested_note(cal, t))
    st.caption(t["data_date"].format(date=f"{max(load_close(labels[p]).index[-1] for p in picked):%Y-%m-%d}"))


if __name__ == "__main__":
    main()
