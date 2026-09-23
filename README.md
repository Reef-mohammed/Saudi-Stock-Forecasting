# Saudi Stock Forecast

**Live app: [saudi-stock-forecasting.streamlit.app](https://saudi-stock-forecasting.streamlit.app)** (Arabic and English, works on phones)

Pick one or more companies listed on the Saudi Exchange (Tadawul) and see three scenarios for
the share price over the next 5 years: a **crash case**, a **likely** case and a **good case**.
Every scenario was tested against what actually happened to Saudi stocks since 2005.

> اختر شركة أو أكثر من الشركات المدرجة في تداول لترى ثلاثة احتمالات لسعر السهم خلال السنوات
> الخمس القادمة: حالة الانهيار، والمتوقع، والحالة الجيدة، مختبرة على بيانات السوق منذ عام 2005.

---

## Why scenarios instead of a single prediction

No model reliably predicts where a stock will be in 5 years, and this project set out to prove
that honestly rather than hide it. Before building the app, I backtested several forecasting
models on **322,000 historical forecasts** across 187 Saudi companies. The simplest possible
model, "the price stays where it is today", was the hardest to beat.

So instead of one number, the app shows a range, and the range is **calibrated**: its width is
set from backtests so the real outcome landed inside it about 9 times out of 10.

## Results

### 1. Point forecasts: nothing beat the naive model

Walk-forward backtest, monthly forecast origins from 2012, each model trained only on data
available at the time. Error is the mean absolute percentage error of the forecast price.

| Horizon | Naive (price stays) | Drift (historical trend) | XGBoost (all stocks, oil, market) | XGBoost direction right |
|---|---|---|---|---|
| 1 week | **3.3%** | 3.3% | 3.3% | 50.3% |
| 1 month | **7.4%** | 7.5% | 7.5% | 51.9% |
| 3 months | **13.2%** | 13.8% | 13.4% | 52.5% |
| 1 year | **29.8%** | 35.2% | 31.3% | 47.2% |

- Extending past trends (drift) made forecasts worse at every horizon.
- A global XGBoost model with price, volatility, market and Brent oil features came close but
  never beat naive on error, and called direction barely better than a coin flip.

### 2. Scenario ranges: calibrated to survive crashes

The 5-year scenarios come from a Monte Carlo simulation that replays random one-month blocks of
each stock's real history (a block bootstrap), which keeps crashes and volatile periods that a
normal distribution would miss. The likely line is flat at today's price, because a
market-growth drift was far less accurate (e.g. 305% vs 73% average error at 5 years).

A range that holds "80% of the time on average" can still fail badly in a crash, when almost
every stock falls at once: for forecasts made just before the 2006 Saudi crash, fewer than half
of real outcomes landed in an average-calibrated range. So the two sides are sized differently:

- **Crash case:** widened until, in **every** start year since 2005, at most 1 in 10 stocks
  ended below it, including the 2006 crash.
- **Good case:** about 1 in 10 stocks ended above it on average. Beating the good case is a
  pleasant surprise, so it is not sized for the most extreme boom.

| Horizon | Forecasts tested | Ended below crash case | Worst start year | Ended above good case |
|---|---|---|---|---|
| 1 year | 10,676 | 0.4% | 9.7% | 10.0% |
| 3 years | 9,186 | 0.7% | 9.7% | 10.0% |
| 5 years | 7,735 | 1.2% | 9.7% | 10.0% |

The app stops at 5 years: 10-year outcomes exist only for forecasts made in 2005–2016, which
overlap so heavily that they cover just one or two independent periods, too few to trust.

## Data

- **History (2001–2020):** [Saudi Stock Exchange (Tadawul)](https://www.kaggle.com/datasets/salwaalzahrani/saudi-stock-exchange-tadawul)
  on Kaggle (CC0).
- **Recent prices (2019 to today):** Yahoo Finance via `yfinance`. The app tops up each company
  with the latest prices when it is opened (cached for 6 hours) and falls back to the stored
  history if Yahoo is unavailable.
- **Merging:** where the two sources overlap, Kaggle history is rescaled to match Yahoo's
  split-adjusted prices (97 of 199 companies needed it).
- **Cleaning:** Tadawul limits daily moves to ±10%, so a one-day move over 25% is either a bad
  tick (dropped when it reverts the next day) or an unadjusted corporate action such as a capital
  reduction or rights issue (earlier history back-adjusted). 26 such breaks were fixed across
  19 companies.
- **Company names:** Arabic names from the Tadawul list on Arabic Wikipedia; English names
  updated for companies renamed since 2020 (e.g. Saudi National Bank, Saudi Awwal Bank).

The final dataset has about 897,000 daily prices for 199 companies, 187 of them still listed.

## Project structure

```
app.py                  Streamlit app (Arabic / English, mobile friendly)
build_dataset.py        Download, merge and clean Kaggle + Yahoo prices -> data/
backtest.py             Walk-forward backtest of naive, drift and XGBoost models
calibrate_scenarios.py  Backtest and calibrate the Monte Carlo crash / likely / good scenarios
export_app_data.py      Copy what the app needs from data/ into app_data/
models/
  base.py               Shared forecaster interface
  baselines.py          Naive and drift models
  xgb.py                Global XGBoost quantile model and features
  montecarlo.py         Block-bootstrap scenario simulator
app_data/               Price history, company list and calibration used by the app
company_names.csv       English and Arabic company names
```

## Run it yourself

```bash
pip install -r requirements-dev.txt

# App only (uses the stored data in app_data/)
streamlit run app.py

# Full pipeline: put the Kaggle CSV in data/raw/ first
python build_dataset.py --kaggle data/raw/Tadawul_stcks.csv --extra-tickers tickers_extra.csv
python backtest.py
python calibrate_scenarios.py
python export_app_data.py
```

## Disclaimer

This project is for illustration and learning only and is not a recommendation to buy or sell
any security. Past performance does not guarantee future results, and actual losses may exceed
the crash case.
