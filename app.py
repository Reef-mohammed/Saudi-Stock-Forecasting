"""
5-Year Saudi Stock Forecast: pick companies, see crash / likely / good cases over the next 5 years.

Run locally:  streamlit run app.py
Reads the stored history in app_data/ (export_app_data.py) and tops it up with recent Yahoo
prices for each company the user opens, cleaned the same way as the dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from build_dataset import PRICE_COLS, adjust_breaks, clean, drop_spikes, fetch_yfinance, merge_sources
from models.montecarlo import MonteCarlo, scale_curve

DATA = Path("app_data")
LIVE_TTL = 6 * 3600               # refetch recent prices from Yahoo at most every 6 hours
LIVE_OVERLAP_DAYS = 365           # overlap with stored history, to detect new splits and rescale
# Current English and Arabic names: Arabic from Tadawul's list on Arabic Wikipedia, English updated
# for companies renamed after the 2020 dataset (e.g. Saudi National Bank, Saudi Awwal Bank)
NAMES = Path("company_names.csv")
MAX_YEARS = 5
MAX_COMPANIES = 5
RLM = chr(0x200F)                 # right-to-left mark
RTL_ISOLATE = "\u2067{}\u2069"    # keeps an Arabic label right-to-left inside left-to-right text
DAYS_PER_MONTH = 30.44
ARABIC_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
                 "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
LTR = "\u200e"                    # keeps "-75%" from being flipped to "75%-" in Arabic text
LTR_ISOLATE = "\u2066{}\u2069"    # keeps "2005–2021" in order inside Arabic text
# Streamlit's own styles left-align headings, captions and lists, so right-align them explicitly.
# Charts stay left-to-right (time runs left to right in both languages).
RTL_CSS = """<style>
.stMainBlockContainer { direction: rtl; }
/* The company dropdown opens outside the main container, so it needs its own direction */
[data-testid="stMultiSelectDropdown"] { direction: rtl; text-align: right; }
.stMainBlockContainer h1, .stMainBlockContainer h2, .stMainBlockContainer h3,
.stMainBlockContainer p, .stMainBlockContainer li, .stMainBlockContainer th,
.stMainBlockContainer td, [data-testid="stCaptionContainer"] { text-align: right !important; }
.stMainBlockContainer ul, .stMainBlockContainer ol { padding-right: 1.5em; padding-left: 0; }
/* Streamlit separates table columns with a right border; mirrored columns need it on the left */
.stMainBlockContainer th, .stMainBlockContainer td {
  border-left: 1px solid rgba(128, 128, 128, 0.2) !important; border-right: none !important; }
.stMainBlockContainer tr > :last-child { border-left: none !important; }
.js-plotly-plot, .js-plotly-plot * { direction: ltr; }
/* Hover box text is Arabic, so lay it out right to left ("7 مايو 2025", not "مايو 2025 7") */
.js-plotly-plot .hoverlayer .hovertext text { direction: rtl; unicode-bidi: plaintext; }
</style>"""
# Plotly's hover mode "x" also labels the axis with an English date; the hover box already shows it
CHART_CSS = """<style>
.js-plotly-plot .hoverlayer .axistext { display: none; }
/* Light shading on the summary table header (faint grey in light mode, where white would vanish) */
.stMainBlockContainer table thead th { background-color: rgba(255, 255, 255, 0.08); font-weight: 700; }
@media (prefers-color-scheme: light) {
  .stMainBlockContainer table thead th { background-color: rgba(0, 0, 0, 0.04); }
}
</style>"""
COLORS = {"crash": "#d64545", "likely": "#2f6fdb", "good": "#2e9d5b", "history": "#555555",
          "band": "rgba(47,111,219,0.12)"}

T = {
    "en": {
        "title": "5-Year Saudi Stock Forecast",
        "subtitle": "Three possibilities for the share price in 5 years (crash, likely and good), based on over 20 years of Tadawul data.",
        "pick": "Choose one or more companies", "placeholder": "Search companies",
        "lang": "العربية",
        "crash": "Crash case", "likely": "Likely", "good": "Good case", "history": "Past price",
        "today": "Today", "company": "Company", "price_today": "Today (SAR)", "sar": "SAR",
        "in_years": "Next 5 years",
        "summary": "Summary", "no_pick": "Choose at least one company to see its scenarios.",
        "what_title": "What do these numbers mean?",
        "what": (
            "- **Likely** is today's price. In our tests, no method predicted the direction of Saudi "
            "stocks better than assuming they stay where they are.\n"
            "- **Crash case** is built to hold even in a severe market crash: in every year we "
            "tested since 2005, including the worst ones, at most 1 in 10 stocks ended below it.\n"
            "- **Good case**: about 1 in 10 stocks ended above it. Beating it is possible, especially "
            "in a boom.\n"
            "- The shaded area is where the price stayed about 9 times out of 10 in our tests."
        ),
        "warning": (
            "**Disclaimer:** This information is for illustration only and is not a recommendation "
            "to buy or sell. Past performance does not guarantee future results, and actual losses "
            "may exceed the crash case."
        ),
        "tested": "Tested on {n:,} past 5-year forecasts made {origins}: {below} ended below the crash case, "
                  "{above} above the good case.",
        "short": "{name} has only {years:.1f} years of history, so its scenarios are less reliable.",
        "no_crash": "{name} is a newer listing, so its own history has no severe market crash. "
                    "Its crash case is widened using how other Saudi stocks fell in past crashes.",
        "data_date": "Last update: {date}",
    },
    "ar": {
        "title": "توقعات الأسهم السعودية لـ 5 سنوات",
        "subtitle": "ثلاثة احتمالات لسعر السهم بعد 5 سنوات: انهيار، متوقع، وجيد، مبنية على أكثر من 20 سنة من بيانات تداول.",
        "pick": "اختر شركة أو أكثر", "placeholder": "ابحث عن شركة",
        "lang": "English",
        "crash": "حالة الانهيار", "likely": "المتوقع", "good": "الحالة الجيدة", "history": "السعر السابق",
        "today": "اليوم", "company": "الشركة", "price_today": "اليوم (ريال)", "sar": "ريال",
        "in_years": "السنوات الخمس القادمة",
        "summary": "ملخص", "no_pick": "اختر شركة واحدة على الأقل لعرض السيناريوهات.",
        "what_title": "ماذا تعني هذه الأرقام؟",
        "what": (
            "- **المتوقع** هو سعر اليوم. في اختباراتنا لم تتفوق أي طريقة في توقع اتجاه الأسهم السعودية "
            "على افتراض بقاء السعر كما هو.\n"
            "- **حالة الانهيار** مصممة لتصمد حتى في انهيارات السوق الحادة: في كل سنة اختبرناها منذ 2005، "
            "بما فيها أسوأ السنوات، انتهى سهم واحد من كل 10 أسهم على الأكثر تحتها.\n"
            "- **الحالة الجيدة**: انتهى سهم واحد تقريباً من كل 10 أسهم فوقها، وتجاوزها ممكن خاصة في فترات الارتفاع.\n"
            "- المنطقة المظللة هي المكان الذي بقي فيه السعر تقريباً 9 مرات من كل 10 في اختباراتنا."
        ),
        "warning": (
            "**تنويه:** المعلومات الواردة لأغراض توضيحية فقط ولا تُعدّ توصية بالشراء أو البيع. "
            "الأداء السابق لا يضمن النتائج المستقبلية، وقد تكون الخسائر الفعلية أكبر من حالة الانهيار."
        ),
        "tested": "تم اختبارها على {n:,} توقع سابق لمدة 5 سنوات بين {origins}: {below} انتهت تحت حالة الانهيار، "
                  "و{above} فوق الحالة الجيدة.",
        "short": "لدى {name} بيانات لمدة {years:.1f} سنوات فقط، لذلك سيناريوهاتها أقل موثوقية.",
        "no_crash": "{name} شركة مدرجة حديثاً، لذلك لا يحتوي تاريخها على انهيار حاد للسوق. "
                    "تم توسيع حالة الانهيار لها بناءً على هبوط الأسهم السعودية الأخرى في الانهيارات السابقة.",
        "data_date": "آخر تحديث: {date}",
    },
}


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_companies() -> pd.DataFrame:
    """Active companies indexed by ticker, with a display label per language: 'Name (1234)'."""
    c = pd.read_csv(DATA / "companies.csv", parse_dates=["first_date", "last_date"])
    c = c.merge(pd.read_csv(NAMES), on="ticker", how="left")
    c["name_en"] = c["name_en"].fillna(c["name"])
    c["name_ar"] = c["name_ar"].fillna(c["name_en"])
    code = " (" + c["ticker"].str.replace(".SR", "", regex=False) + ")"
    c["label_en"] = c["name_en"] + code
    # Isolate each Arabic label as right-to-left, so "(1120)" stays after the name even where the
    # surrounding text is left-to-right (the dropdown list, chart titles)
    c["label_ar"] = (c["name_ar"] + code).map(RTL_ISOLATE.format)
    return c.set_index("ticker")


@st.cache_data
def load_calibration() -> dict:
    return json.loads((DATA / "scenario_calibration.json").read_text())


@st.cache_data
def stored_close(ticker: str) -> pd.Series:
    p = pd.read_parquet(DATA / "prices.parquet", filters=[("ticker", "==", ticker)])
    return p.set_index("date")["close"].astype(float).sort_index()


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def load_close(ticker: str) -> pd.Series:
    """Stored history topped up with recent Yahoo prices; the stored history alone if Yahoo fails."""
    stored = stored_close(ticker)
    try:
        start = stored.index[-1] - pd.Timedelta(days=LIVE_OVERLAP_DAYS)
        recent = clean(fetch_yfinance([ticker], f"{start:%Y-%m-%d}"))
        if recent.empty:
            return stored
        hist = stored.rename("close").reset_index().assign(ticker=ticker, volume=np.nan)
        hist["date"] = hist["date"].astype("datetime64[ns]")      # match Yahoo's date type for the merge
        recent["date"] = recent["date"].astype("datetime64[ns]")
        for c in PRICE_COLS:
            hist[c] = hist["close"]
        # Rescales stored history if Yahoo has adjusted for a split since it was saved
        merged = adjust_breaks(drop_spikes(merge_sources(hist, recent)))
        return merged.set_index("date")["close"].sort_index()
    except Exception as e:                        # network down, Yahoo rate limit, format change...
        logging.warning("Live prices failed for %s, using stored history: %s", ticker, e)
        return stored


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
    return t["tested"].format(n=r["n"], origins=LTR_ISOLATE.format(r["origins"].replace("-", "–")),
                              below=LTR_ISOLATE.format(f"{r['below_worst']:.0%}"),
                              above=LTR_ISOLATE.format(f"{r['above_best']:.0%}"))


# ----------------------------------------------------------------------------- chart
def chart(name: str, close: pd.Series, sc: pd.DataFrame, t: dict, rtl: bool) -> go.Figure:
    hist = close[close.index >= close.index[-1] - pd.DateOffset(years=2)]
    today = pd.DataFrame({"date": [close.index[-1]], "crash": [close.iloc[-1]],
                          "likely": [close.iloc[-1]], "good": [close.iloc[-1]]})
    sc = pd.concat([today, sc], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name=t["history"], hoverinfo="skip",
                             line=dict(color=COLORS["history"], width=1.5)))
    fig.add_trace(go.Scatter(x=sc["date"], y=sc["good"], name=t["good"], hoverinfo="skip",
                             line=dict(color=COLORS["good"], width=2)))
    fig.add_trace(go.Scatter(x=sc["date"], y=sc["crash"], name=t["crash"], fill="tonexty",
                             fillcolor=COLORS["band"], hoverinfo="skip",
                             line=dict(color=COLORS["crash"], width=2)))
    fig.add_trace(go.Scatter(x=sc["date"], y=sc["likely"], name=t["likely"], hoverinfo="skip",
                             line=dict(color=COLORS["likely"], width=2, dash="dash")))

    # One invisible line carries the whole hover box. Streamlit's Plotly has no Arabic month
    # names, so the dates are written out here rather than formatted by Plotly.
    def date_text(d: pd.Timestamp, with_day: bool) -> str:
        month = ARABIC_MONTHS[d.month - 1] if rtl else f"{d:%b}"
        return f"{d.day} {month} {d.year}" if with_day else f"{month} {d.year}"

    def line(key: str, value: float) -> str:
        color = "#dddddd" if key == "history" else COLORS[key]    # grey line colour is too dark here
        return f"<span style='color:{color}'>{t[key]}: {value:.2f} {t['sar']}</span>"

    past = [f"<b>{date_text(d, True)}</b><br>{line('history', v)}" for d, v in hist.items()]
    future = [f"<b>{date_text(r.date, False)}</b><br>{line('likely', r.likely)}<br>"
              f"{line('crash', r.crash)}<br>{line('good', r.good)}" for r in sc.iloc[1:].itertuples()]
    fig.add_trace(go.Scatter(x=[*hist.index, *sc["date"].iloc[1:]], y=[*hist.values, *sc["likely"].iloc[1:]],
                             customdata=past + future, hovertemplate="%{customdata}<extra></extra>",
                             mode="lines", line=dict(width=0), showlegend=False))

    side = dict(x=1, xanchor="right") if rtl else dict(x=0, xanchor="left")
    fig.update_layout(title=dict(text=name, font=dict(size=16), **side), height=360,
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x",
                      hoverlabel=dict(align="right" if rtl else "left", bgcolor="#1b1d24",
                                      bordercolor="#444444", font=dict(color="#f0f0f0")),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, **side),
                      yaxis_title="SAR", dragmode=False)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                     spikedash="dot", spikecolor="#888888")
    return fig


# ----------------------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="5-Year Saudi Stock Forecast", page_icon="📈", layout="centered")
    if "lang" not in st.session_state:
        st.session_state.lang = "ar"
    t = T[st.session_state.lang]
    st.markdown(CHART_CSS, unsafe_allow_html=True)
    if st.session_state.lang == "ar":
        st.markdown(RTL_CSS, unsafe_allow_html=True)

    top = st.columns([4, 1])
    top[0].title(t["title"])
    if top[1].button(t["lang"], use_container_width=True):
        st.session_state.lang = "en" if st.session_state.lang == "ar" else "ar"
        st.rerun()
    st.caption(t["subtitle"])

    companies = load_companies()
    cal = load_calibration()
    lang = st.session_state.lang
    labels = companies[f"label_{lang}"].sort_values()
    # Streamlit keeps a widget's option labels per key, so each language gets its own picker;
    # the chosen tickers are carried over when the language switches.
    key = f"picked_{lang}"
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get("picked", ["1120.SR", "2222.SR"])
    # At most 5: each company adds a chart and a live price download
    picked = st.multiselect(t["pick"], labels.index, key=key, format_func=labels.get,
                            max_selections=MAX_COMPANIES, select_all=False,
                            placeholder=t["placeholder"])
    st.session_state.picked = picked

    if not picked:
        st.info(t["no_pick"])
        st.warning(t["warning"])
        return
    st.warning(t["warning"])

    rows, figs, notes = [], [], []
    for ticker in picked:
        label, name = labels[ticker], companies.loc[ticker, f"name_{lang}"]
        close = load_close(ticker)
        sc = scenarios(ticker, close.index[-1])
        end, now = sc.iloc[-1], close.iloc[-1]
        # Percent changes keep the table narrow enough for a phone; prices are on the charts
        rows.append({t["company"]: label, t["price_today"]: f"{now:.2f}",
                     **{t[k]: pct(end[k] / now - 1) for k in ("crash", "likely", "good")}})
        # Chart titles are drawn as SVG, where phones ignore the isolate marks in the label; a
        # right-to-left mark (RLM) on each side keeps "(1120)" after the name there
        title = f"{RLM}{name} ({ticker[:4]}){RLM}" if lang == "ar" else label
        figs.append(chart(title, close, sc, t, rtl=lang == "ar"))
        hist_years = (close.index[-1] - close.index[0]).days / 365.25
        if hist_years < 5:
            notes.append(t["short"].format(name=name, years=hist_years))
        elif close.index[0] > pd.Timestamp("2006-01-01"):
            notes.append(t["no_crash"].format(name=name))

    st.subheader(t["summary"] + " · " + t["in_years"])
    st.table(pd.DataFrame(rows).set_index(t["company"]))
    for note in notes:
        st.caption("⚠️ " + note)

    for fig in figs:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander(t["what_title"], expanded=True):
        st.markdown(t["what"])
        st.caption(tested_note(cal, t))
    last = max(load_close(p).index[-1] for p in picked)
    st.caption(t["data_date"].format(date=LTR_ISOLATE.format(f"{last:%Y-%m-%d}")))


if __name__ == "__main__":
    main()
