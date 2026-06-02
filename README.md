# Hybrydowy Model Krypto (Crypto Hybrid Model)

Trójwarstwowy framework badawczy dla Bitcoin i Ethereum, który łączy statystyczne modelowanie zmienności (GARCH) z uczeniem maszynowym do prognozowania log-zwrotów na następny dzień, ewaluacji systematycznych strategii Long/Flat oraz symulacji ryzyka rynkowego metodą Monte Carlo.

---

## Architektura

```text
Warstwa 0 — Dane
  data/data.py          CryptoDataFetcher
                        Pobieranie danych (yfinance) → log-zwroty, roczna zmienność, wolumen
                        Pamięć podręczna na dysku (parquet / CSV fallback)

Warstwa 1 — Statystyka
  statistical_analysis.py   CryptoVolatilityModel
                            Test stacjonarności ADF
                            Warunkowa zmienność GARCH(1,1) (annualizowana)

Warstwa 2 — Uczenie Maszynowe
  mlbase.py             Wspólne stałe i struktura FoldMetric
  features.py           FeatureMixin – inżynieria cech (bez wycieku danych)
  backtest.py           BacktestMixin – ewaluacja strategii Long/Flat
  persistence.py        PersistenceMixin – zapis/odczyt modeli (joblib)
  model.py              CryptoMLPredictor – trenowanie, predykcja, strojenie hiperparametrów
  report.py             Formatowanie konsolowe i przepływ main()
  finance_metrics.py    Czyste funkcje ryzyka/zwrotu (Sharpe, Sortino,
                        max drawdown, VaR, CVaR, kalibracja walk-forward)
  ml_forecasting.py     Cienka fasada zachowująca oryginalny punkt wejścia

Warstwa 3 — Zarządzanie Ryzykiem i Symulacja Rynku
  risk_management.py    CryptoMonteCarloSimulator
                        Zwektoryzowane ścieżki geometrycznego ruchu Browna (GBM)
                        zasilane dryfem z ML (mu) oraz zmiennością z GARCH (sigma)
                        VaR / CVaR (95% i 99%), alokacja Sharpe/Kelly, wykresy
```

---

## Symulacje Monte Carlo

W oparciu o wyjścia z modelu ML (przewidywany zwrot) i modelu statystycznego (przewidywana zmienność), warstwa zarządzania ryzykiem symuluje 10 000 możliwych scenariuszy cenowych w wybranym horyzoncie czasowym (np. 30 dni). Poniższe wykresy przedstawiają wygenerowane ścieżki cenowe oraz rozkład ryzyka z zaznaczonym Value at Risk (VaR).

### Bitcoin (BTC-USD)
![Symulacja Monte Carlo BTC](montecarlo_BTC_USD.png)

### Ethereum (ETH-USD)
![Symulacja Monte Carlo ETH](montecarlo_ETH_USD.png)

---

## Funkcjonalności

| Kategoria | Szczegóły |
|---|---|
| **Dane** | Codzienne dane OHLCV dla BTC-USD i ETH-USD z yfinance; opcjonalny szybki cache na dysku |
| **Statystyka** | Test ADF, model GARCH(1,1) z mechanizmem ponawiania zbieżności, dynamiczna annualizacja |
| **Cechy** | Opóźnienia log-zwrotów (lags 1-7d), stosunki średnich ruchomych SMA, RSI-14, historyczna zmienność krocząca, zmienność GARCH, wskaźniki wolumenu, kontekst międzyrynkowy BTC |
| **Model** | XGBRegressor (preferowany) lub RandomForestRegressor; walidacja TimeSeriesSplit |
| **Strojenie** | Opcjonalny RandomizedSearchCV z TimeSeriesSplit (chroniący przed wyciekiem danych) |
| **Backtest** | Strategia Long/Flat uwzględniająca narzucone koszty transakcyjne (w punktach bazowych) |
| **Progowanie** | Grid-search (optymalizacja pod Sortino) + bezpieczna zagnieżdżona kalibracja walk-forward |
| **Metryki** | MAE, R², trafność kierunkowa, Sharpe, Sortino, max drawdown, VaR, CVaR, win rate, całkowity zysk |
| **Werdykt** | Klasyfikacja oparta na regułach: PEŁNA PRZEWAGA / DEFENSYWNA PRZEWAGA / BRAK PRZEWAGI |
| **Zapis** | Generowanie plików modeli joblib i pliku metadanych metadata.json |
| **Symulacja** | Zwektoryzowane GBM Monte Carlo (np. 10k ścieżek) połączone bezpośrednio z parametrami z ML i GARCH |
| **Ryzyko ogona** | Value at Risk (VaR) oraz Expected Shortfall (CVaR) na poziomach 95% i 99% z terminalnego rozkładu |
| **Alokacja** | Podział portfela BTC/ETH na bazie kryterium Sharpe'a lub Kelly'ego |

---

## Szybki Start

### 1. Utwórz i aktywuj środowisko wirtualne

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate     # Linux / macOS
```

### 2. Zainstaluj zależności

```powershell
pip install -r requirements.txt
```

> **Uwaga dla Windows**: jeśli instalacja pakietu `xgboost` zakończy się błędem, zainstaluj darmowy dodatek [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### 3. Uruchom rurociąg (pipeline)

```powershell
# Z głównego folderu projektu:
python main.py
# lub używając oryginalnego punktu wejścia:
python ml_forecasting.py
```

Podczas pierwszego uruchomienia dane zostaną pobrane z Yahoo Finance i zapisane w folderze `data_cache/`. Każde kolejne uruchomienie wczyta je w ułamek sekundy z dysku. 

Aby uruchomić wyłącznie analizę ryzyka (warstwa 3) z opcją generowania wykresów:
```powershell
python risk_management.py
```

---

## Raport i Wyjście (Output)

Standardowe uruchomienie `main.py` generuje w konsoli 8 sekcji:

1. **Layer 2 ML Forecasting Metrics** — metryki MAE i R² (dla każdego foldu i uśrednione)
2. **Feature Importances** — waga i wpływ cech na decyzyjność modelu XGBoost
3. **Financial Metrics** — Long/Flat w porównaniu z klasycznym Buy & Hold (przy stałym progu wejścia)
4. **Investment Verdict** — werdykt oparty na metrykach z dołączonymi zastrzeżeniami
5. **Calibrated Threshold** — rzut okiem na zoptymalizowany próg (dane in-sample; granica optymistyczna)
6. **Walk-Forward Validation** — bezwzględna walidacja bez wycieku danych z przyszłości
7. **Next-Day Forecasts** — prognozowany log-zwrot na następną sesję giełdową
8. **Monte Carlo Risk Simulation** — VaR/CVaR oraz sugestie alokacji (uruchomienie Warstwy 3 na najnowszych danych)

---

## Odniesienia do modułów

| Moduł | Główne API |
|---|---|
| `data/data.py` | `CryptoDataFetcher(tickers, start, end, interval, cache_dir)` |
| `statistical_analysis.py` | `CryptoVolatilityModel(returns, trading_days)` |
| `model.py` | `CryptoMLPredictor(tickers, start, end, ..., cache_dir)` |
| `finance_metrics.py` | `summarize`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `walk_forward_threshold`, … |
| `risk_management.py` | `CryptoMonteCarloSimulator(current_price, expected_return, volatility)`, `suggest_allocation` |

**Przykład użycia z poziomu kodu:**
```python
from model import CryptoMLPredictor
from risk_management import CryptoMonteCarloSimulator

predictor = CryptoMLPredictor(cache_dir="data_cache")
predictor.train_all()

# Prognoza zwrotu na następny dzień rynkowy
btc_mu = predictor.predict_next_day_return("BTC-USD")

# Walidacja i uczciwy backtest
result = predictor.walk_forward_threshold("BTC-USD")
print(result["verdict"])

# Warstwa 3: Symulacja ryzyka napędzana danymi z ML oraz GARCH
price = float(predictor.close_prices["BTC-USD"].dropna().iloc[-1])
sigma = float(predictor.garch_features["BTC-USD_GARCH_Vol"].dropna().iloc[-1])
sim = CryptoMonteCarloSimulator(price, btc_mu, sigma, ticker="BTC-USD")

sim.run_simulation(horizon_days=30, num_simulations=10_000)
print(sim.calculate_risk_metrics().report())

# Persystencja (zapisanie wytrenowanych modeli na dysk)
predictor.save_models("models_artifacts")
```

---

## Przewodnik Interpretacji Wyników

| Metryka / Pojęcie | Co to oznacza |
|---|---|
| **R² < 0** | Model radzi sobie słabiej niż przewidywanie średniej. Jest to **oczekiwane i normalne** w predykcji dziennych cen wirtualnych aktywów. Stosunek sygnału do szumu jest krytycznie niski. |
| **Trafność kierunkowa ~49%** | Rzut monetą. Model nie uczy się przepowiadania samego kierunku (co jest bliskie niemożliwości). Model znajduje przewagę ucząc się tego *kiedy rynek będzie mocno zmienny*. |
| **Mniejsza zmienność, mniejszy drawdown** | System siedząc znaczny czas w gotówce obniża ryzyko i straty. To jest faktyczna główna zaleta działania tego algorytmu. |
| **PEŁNA PRZEWAGA** | Strategia algorytmiczna pokonuje benchmark ("Kup i trzymaj") na wykresie zwrotu, jak i w metrykach skorygowanych o ryzyko. |
| **DEFENSYWNA PRZEWAGA** | Algorytm dostarcza mniejszych obsunięć kapitału i ma lepsze wskaźniki Sharpe/Sortino, lecz wygenerował mniejszy zysk całkowity. "Bezpieczniej, ale mniej zyskownie". |
| **BRAK PRZEWAGI** | Wyniki są we wszystkich parametrach gorsze od standardowego trzymania krypto. |

> Zawsze polegaj przede wszystkim na wynikach z bloku **Walk-Forward Validation**. Moduły wchodzące w jego skład oceniają skuteczność, udostępniając do treningu i kalibracji progu wyłącznie historyczne punkty (symulując idealnie zachowanie "na żywo").

---

## Wymagania

- Python w wersji 3.10+
- Biblioteki wymienione w `requirements.txt`
- Połączenie internetowe na potrzeby pierwszego pobrania paczki danych (następnie narzędzie działa również w wariancie offline, o ile pamięć podręczna została zbudowana).

---

## Zastrzeżenie prawne (Disclaimer)

Zbudowany algorytm, rurociąg badawczy oraz wszystkie dostarczane przez niego wyniki mają charakter wyłącznie badawczy, statystyczny i edukacyjny. W żadnym wypadku projekt nie stanowi on doradztwa inwestycyjnego, finansowego ani zachęty do obrotu prawdziwymi środkami. Algorytm demonstruje koncepcje Data Science. Należy pamiętać, że wyniki osiągnięte w symulacjach i backtestach nie dają cienia gwarancji podobnego działania w historycznej i nienotowanej jeszcze przyszłości.
