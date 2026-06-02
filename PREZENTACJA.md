# Skrypt prezentacji: Hybrydowy Model Krypto

**Czas trwania:** ok. 20–25 minut (+ 5 min na pytania)  
**Odbiorca:** wykładowca, rekruter, jury projektu lub zespół techniczny  
**Demonstracja na żywo:** `python main.py` (wymaga internetu przy pierwszym uruchomieniu lub cache w `data_cache/`)

---

## Spis treści

1. [Wstęp i cel projektu](#1-wstęp-i-cel-projektu-2-min)
2. [Architektura i przepływ danych](#2-architektura-i-przepływ-danych-3-min)
3. [Wprowadzenie teoretyczne — statystyka](#3-wprowadzenie-teoretyczne--statystyka-5-min)
4. [Warstwa ML — jak działa algorytm](#4-warstwa-ml--jak-działa-algorytm-5-min)
5. [Warstwa ryzyka — Monte Carlo](#5-warstwa-ryzyka--monte-carlo-3-min)
6. [Przegląd kodu — co gdzie jest](#6-przegląd-kodu--co-gdzie-jest-3-min)
7. [Interpretacja wyników i ocena skuteczności](#7-interpretacja-wyników-i-ocena-skuteczności-5-min)
8. [Ograniczenia i dalszy rozwój](#8-ograniczenia-i-dalszy-rozwój-2-min)
9. [Zamknięcie i pytania](#9-zamknięcie-i-pytania)
10. [Ściąga — najważniejsze liczby z Twojego uruchomienia](#ściąga--najważniejsze-liczby-z-twojego-uruchomienia)

---

## 1. Wstęp i cel projektu (~2 min)

### Co powiedzieć

> „Prezentuję **hybrydowy model analityczny** dla Bitcoina i Ethereum. Projekt nie jest gotowym botem tradingowym — to **framework badawczy**, który łączy trzy podejścia: klasyczną ekonometrię zmienności, uczenie maszynowe oraz symulację ryzyka Monte Carlo.
>
> Główny problem, który rozwiązujemy: **dzienne zwroty krypto są trudne do przewidzenia kierunkowo** (są blisko białego szumu), ale **zmienność się klastruje** — okresy spokoju i burzy na rynku da się modelować. Dlatego model uczy się głównie kontekstu ryzyka, a strategia inwestycyjna opiera się na **decyzji Long/Flat** (w rynku vs w gotówce), a nie na ciągłym obstawianiu kierunku.”

### Kluczowe hasła na slajdzie

- Dane: Yahoo Finance (`yfinance`), BTC-USD, ETH-USD, interwał dzienny
- Trzy warstwy: Statystyka → ML → Ryzyko
- Uczciwa walidacja: `TimeSeriesSplit`, walk-forward bez wycieku danych

### Demo (opcjonalnie na początku)

```powershell
cd crypto_hybrid_model
python main.py
```

---

## 2. Architektura i przepływ danych (~3 min)

### Co powiedzieć

> „Dane wchodzą przez moduł `data/data.py`. Klasa `CryptoDataFetcher` pobiera OHLCV, liczy **log-zwroty** — bo sumują się addytywnie w czasie i lepiej nadają się do modeli statystycznych niż procenty arytmetyczne.
>
> Z log-zwrotów budujemy **cechy** w `features.py`: lagi, RSI, stosunki do średnich kroczących, zmienność realizowana i **zmienność warunkowa GARCH** z warstwy 1.
>
> Model XGBoost w `model.py` przewiduje **log-zwrot następnego dnia**. Predykcja plus próg decyzyjny trafiają do **backtestu** w `backtest.py` i `finance_metrics.py`.
>
> Na końcu `risk_management.py` bierze **ostatnią cenę**, **prognozę μ z ML** i **σ z GARCH** i uruchamia **10 000 ścieżek Monte Carlo** na 30 dni.”

### Schemat do narysowania na tablicy / slajdzie

```text
yfinance → log-zwroty → [ADF + GARCH σ] → cechy + target → XGBoost → sygnał Long/Flat
                                                              ↓
                                         backtest / walk-forward / werdykt
                                                              ↓
                                         Monte Carlo (μ, σ) → VaR, CVaR, wykresy
```

### Ważna zasada projektu

> **Brak wycieku danych (data leakage):** target to zwrot *następnego* dnia (`shift(-1)`), wszystkie cechy używają tylko przeszłości; w walk-forward próg strategii kalibrujemy wyłącznie na danych *sprzed* każdego okna testowego.

---

## 3. Wprowadzenie teoretyczne — statystyka (~5 min)

### 3.1 Log-zwroty

**Wzór:**

\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

**Co powiedzieć:**

> „Log-zwrot to przybliżenie procentowej zmiany ceny, ale matematycznie wygodniejsze: zwroty z kolejnych dni można dodawać, co odpowiada zwrotowi wielodniowemu. W kodzie: `data/data.py`, metoda `log_returns()`.”

### 3.2 Stacjonarność i test ADF

**Cel:** sprawdzić, czy szereg zwrotów nie ma jednostki korzenia (trendu w wariancji średniej), co jest typowe dla *cen*, ale **zwroty** często są bliżej stacjonarności.

**Co powiedzieć:**

> „Używamy **testu Augmented Dickey-Fuller (ADF)**. Hipoteza zerowa: szereg jest niestacjonarny. Niski p-value → odrzucamy H0 → szereg traktujemy jako stacjonarny. To uzasadnia dalsze modele na zwrotach, nie na surowych cenach. Implementacja: `statistical_analysis.py`, `check_stationarity()`.”

### 3.3 Model GARCH(1,1)

**Intuicja:** wariancja dzisiejsza zależy od wczorajszej zmienności i wczorajszego szoku — stąd **klastry zmienności** (okresy paniki i spokoju).

**Wzór (warunkowa wariancja):**

\[
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
\]

**Annualizacja (krypto 24/7):**

\[
\sigma_{\text{roczna}} = \sigma_{\text{dzienną}} \times \sqrt{365}
\]

**Co powiedzieć:**

> „GARCH(1,1) szacuje **zmienną w czasie wariancję** zwrotów. W projekcie dopasowujemy go biblioteką `arch`, z obsługą problemów zbieżności optymalizatora (ponowne dopasowanie z większą liczbą iteracji). Wynik trafia do kolumny `{ticker}_GARCH_Vol` i staje się cechą `garch_vol` dla modelu ML oraz parametrem σ w symulacji Monte Carlo.”

### 3.4 Co GARCH daje, a czego nie daje

| GARCH | Nie robi |
|--------|----------|
| Szacuje ryzyko / zmienność | Nie przewiduje kierunku ceny |
| Wspiera filtrowanie pozycji (wysoka σ → ostrożność) | Nie gwarantuje zysku |

---

## 4. Warstwa ML — jak działa algorytm (~5 min)

### 4.1 Problem uczenia

**Zmienna docelowa (Y):** log-zwrot **następnego** dnia:

```python
frame[TARGET_COLUMN] = returns.shift(-1)  # features.py
```

**Co powiedzieć:**

> „To regresja: przewidujemy liczbę (oczekiwany log-zwrot), nie klasę ‚w górę / w dół’. Z tej liczby budujemy strategię: jeśli prognoza jest powyżej progu → Long, inaczej → gotówka (Flat).”

### 4.2 Zestaw cech (feature engineering)

| Grupa cech | Przykłady | Sens |
|------------|----------|------|
| Pamięć krótkoterminowa | `return_lag_1` … `return_lag_7` | Autokorelacja, momentum |
| Trend (stacjonarne) | `price_to_sma_7`, `sma_7_to_21` | Pozycja względem średnich — bez poziomów cen |
| Momentum | `rsi_14` | Siła trendu |
| Zmienność | `rolling_vol_7/21`, `garch_vol` | Ryzyko — **najważniejsze w feature importance** |
| Płynność | `volume_change`, `volume_ratio_21` | Aktywność rynku |
| Kontekst BTC | `btc_return`, `btc_garch_vol` (dla ETH) | Współruchomość |

**Co powiedzieć:**

> „Świadomie **nie podajemy surowej ceny** do drzew — drzewa nie ekstrapolują poza zakres treningowy. Zamiast tego stosunki i zwroty. Warstwa GARCH wnosi informację o reżimie zmienności, którą sam zwrot dzienny słabiej niesie.”

### 4.3 Algorytm: XGBoost (XGBRegressor)

**Co powiedzieć:**

> „Używamy **XGBoost** — gradient boosting na drzewach decyzyjnych. Dlaczego: dobra jakość na tabelarycznych cechach finansowych, obsługa nieliniowości, `feature_importances_` do interpretacji. Gdy XGBoost nie jest zainstalowany, jest fallback na **Random Forest**.
>
> Model nie jest siecią neuronową — przy ~1500 obserwacjach dziennych i ~18 cechach drzewa są często stabilniejsze i mniej podatne na przeuczenie niż głębokie sieci.”

**Hiperparametry domyślne (kod `model.py`):**

- `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`
- Cel: `reg:squarederror` (MSE)

### 4.4 Walidacja czasowa

**Co powiedzieć:**

> „Zwykły losowy podział train/test **łamie chronologię** i daje sztucznie dobre wyniki (wyciek z przyszłości). Stosujemy **`TimeSeriesSplit` z 5 podziałami**: w każdym foldzie trenujemy na przeszłości, testujemy na późniejszym odcinku. Metryki MAE i R² raportujemy per fold i uśredniamy.
>
> Predykcje z foldów testowych składamy w jedną serię **out-of-sample** — na niej liczymy metryki finansowe i walk-forward.”

### 4.5 Metryki ML — jak czytać

| Metryka | Dobrze | W naszym projekcie (typowo) |
|---------|--------|---------------------------|
| **MAE** | Niższe = mniejszy błąd bezwzględny | Akceptowalne jako skala błędu |
| **R²** | 1.0 = idealnie, 0 = jak średnia, **< 0 = gorzej niż średnia** | Często **ujemne** na wszystkich foldach |
| **Trafność kierunkowa** | > 50% = przewaga kierunkowa | **~49%** ≈ rzut monetą |

**Co powiedzieć:**

> „Ujemne R² nie oznacza błędu w kodzie — oznacza, że **dzienny kierunek krypto jest praktycznie nieprzewidywalny** na tych cechach. Wartość modelu nie leży w magicznej prognozie wzrostu, tylko w **filtrowaniu ekspozycji** przy wysokim progu sygnału.”

---

## 5. Warstwa ryzyka — Monte Carlo (~3 min)

### 5.1 Geometric Brownian Motion (GBM)

**Wzór dyskretny (dzienny krok):**

\[
S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right], \quad \Delta t = \frac{1}{365}
\]

| Symbol | Źródło w projekcie |
|--------|-------------------|
| \(S_0\) | Ostatnia cena zamknięcia |
| \(\mu\) | Roczny dryf z prognozy ML (dzienny log-zwrot × 365) |
| \(\sigma\) | Roczna zmienność GARCH |
| \(Z\) | Losowa zmienna standardowa (10 000 ścieżek) |

**Co powiedzieć:**

> „Monte Carlo nie ‚poprawia’ modelu ML — **propaguje niepewność** w przód. Dostajemy rozkład cen za 30 dni i z niego liczymy **VaR** (kwantyl strat) i **CVaR / Expected Shortfall** (średnia strata w najgorszym ogonie).”

### 5.2 VaR i CVaR — interpretacja

| Pojęcie | Znaczenie dla inwestora |
|---------|-------------------------|
| **VaR 95%** | W 95% symulacji strata nie była gorsza niż ten próg (5% najgorszych scenariuszy jest poza VaR) |
| **CVaR 95%** | Średnia strata w tych najgorszych 5% przypadków — „jak bardzo boli, gdy już jest źle” |

**Znak w raporcie:** wartości ujemne = strata (zgodnie z `finance_metrics.py`).

### 5.3 Wykresy w README

> „Lewy panel: próbka 100 ścieżek + **mediana**. Prawy panel: histogram cen końcowych z liniami **VaR 95% i 99%**. Pliki: `montecarlo_BTC_USD.png`, `montecarlo_ETH_USD.png`.”

---

## 6. Przegląd kodu — co gdzie jest (~3 min)

### Mapa plików (do pokazania w IDE)

| Plik | Odpowiedzialność |
|------|------------------|
| `data/data.py` | Pobieranie, cache, log-zwroty |
| `statistical_analysis.py` | ADF, GARCH, wykres zmienności |
| `features.py` | Budowa macierzy cech |
| `model.py` | Trening XGBoost, predykcja |
| `backtest.py` | Strategia, optymalizacja progu, walk-forward |
| `finance_metrics.py` | Sharpe, Sortino, drawdown, VaR, werdykt |
| `risk_management.py` | Monte Carlo, alokacja |
| `report.py` | Pipeline `main()` — łączy wszystko |
| `main.py` / `ml_forecasting.py` | Punkty wejścia |

### Fragmenty kodu warto pokazać na ekranie

**1. Brak wycieku — target:**

```python
# features.py
frame[TARGET_COLUMN] = returns.shift(-1)
```

**2. GARCH jako cecha:**

```python
# features.py
frame["garch_vol"] = garch
```

**3. Walk-forward — próg tylko z przeszłości:**

```python
# backtest.py (uproszczenie)
calib_true = y_true_all.iloc[:start]
threshold = fin.best_threshold(calib_true, calib_pred, ...)
```

**4. Integracja Layer 3 w raporcie:**

```python
# report.py
mu = predictor.predict_next_day_return(ticker)
sigma = predictor.garch_features[f"{ticker}_GARCH_Vol"].dropna().iloc[-1]
simulator = CryptoMonteCarloSimulator(price, mu, sigma, ticker=ticker)
```

---

## 7. Interpretacja wyników i ocena skuteczności (~5 min)

### 7.1 Sekcje raportu z `python main.py`

| # | Sekcja | Co oceniamy | Czy ufamy? |
|---|--------|-------------|------------|
| 1 | Metryki ML (MAE, R²) | Jakość dopasowania regresji | Tak, ale R² nie jest celem biznesowym |
| 2 | Feature importances | Co model „widzi” | Tak — zmienność dominuje |
| 3 | Finanse (próg 0) | Naiwna strategia | **Słaba** — dużo fałszywych wejść |
| 4 | Werdykt (próg 0) | Opis vs buy&hold | Informacyjnie |
| 5 | Próg kalibrowany (Sortino) | Optymistyczna górna granica | **Z przestroga** — ten sam OOS |
| 6 | **Walk-forward** | Uczciwy test | **TAK — główna metryka** |
| 7 | Prognoza next-day | μ na jutro | Punktowa, duża niepewność |
| 8 | Monte Carlo | VaR/CVaR, alokacja | Ryzyko forward-looking |

### 7.2 Strategia Long/Flat

**Reguła:**

- Jeśli `y_pred > próg` → pozycja Long (pełna ekspozycja)
- W przeciwnym razie → Flat (gotówka, zwrot 0)
- Przy zmianie pozycji: koszt `cost_bps` (domyślnie 10 bp)

**Co powiedzieć:**

> „Strategia jest **defensywna**: przy wysokim progu jest w rynku mały procent czasu (`time_in_market` ~2–10%). Zarabia na **unikaniu** dużych obsunięć, nie na przewidywaniu każdego wzrostu.”

### 7.3 Werdykty inwestycyjne

| Werdykt | Znaczenie | Przykład z projektu |
|---------|-----------|---------------------|
| **PEŁNA PRZEWAGA** | Lepszy zwrot **i** lepsze metryki ryzyka vs buy&hold | ETH w okresie gdy benchmark spadł |
| **DEFENSYWNA PRZEWAGA** | Lepsze Sharpe/Sortino/drawdown, **gorszy** zwrot całkowity | BTC: mniejsze ryzyko, mniej zysku niż HODL |
| **BRAK PRZEWAGI** | Brak sensownej przewagi | — |

**Zawsze dodaj zastrzeżenie:**

> „Trafność kierunkowa ~50% — brak przewagi predykcyjnej kierunku; korzyść z unikania ryzyka; wynik **zależny od reżimu** (w długiej hossie Long/Flat przegrywa z buy&hold).”

### 7.4 Metryki finansowe — ściąga

| Metryka | Wzór / sens | Wyżej = lepiej? |
|---------|-------------|-----------------|
| **Sharpe** | (zwrot − rf) / zmienność | Tak |
| **Sortino** | jak Sharpe, ale tylko zmienność „złych” zwrotów | Tak |
| **Max drawdown** | Najgłębszy spadek od szczytu equity | Mniej ujemny = lepiej |
| **VaR / CVaR** | Ogony rozkładu strat | Mniej ekstremalna strata |
| **Win rate** | Ułamek dni z dodatnim zwrotem strategii | Przy Long/Flat często niski — **nie mylić z trafnością kierunku** |

### 7.5 Checklist oceny skuteczności (co powiedzieć na koniec)

1. **Czy R² i trafność kierunkowa są realistyczne?** (~49%, R² < 0) → tak, rynek jest trudny.
2. **Czy walk-forward pokazuje sensowną redukcję ryzyka?** → porównaj `ann_volatility` i `max_drawdown` strategii vs benchmark.
3. **Czy wyższy zwrot strategii nie wynika tylko z okresu testowego?** → ETH vs BTC pokazują różne reżimy.
4. **Czy kalibracja progu na tym samym OOS nie jest zbyt optymistyczna?** → tak — dlatego walk-forward jest ważniejszy.
5. **Czy Monte Carlo jest spójne z GARCH/ML?** → σ i μ wchodzą wprost z warstw 1 i 2.

### 7.6 Przykładowa narracja wyników (wg typowego uruchomienia)

> „Dla **BTC** walk-forward: strategia ma **niższą zmienność roczną** i **płytszy drawdown** niż buy&hold, ale **niższy całkowity zwrot** — werdykt **DEFENSYWNA PRZEWAGA**. To profil ‚bezpieczniej, mniej zysku’.
>
> Dla **ETH** benchmark w okresie testowym był **silnie ujemny**, strategia utrzymała niewielki plus dzięki długiemu pobytowi w gotówce — stąd **PEŁNA PRZEWAGA**, ale z wyraźnym zastrzeżeniem o **zależności od reżimu**.
>
> **Feature importances** konsekwentnie wskazują `rolling_vol_7` i `garch_vol` — model słucha ryzyka, nie ‚magicznej’ ceny.”

---

## 8. Ograniczenia i dalszy rozwój (~2 min)

### Ograniczenia (szczerze)

- Dane dzienne z Yahoo — opóźnienia, brak order book
- GARCH(1,1) — uproszczenie; nie modeluje skośności i grubych ogonów (rozszerzenie: GJR-GARCH, t-Student)
- XGBoost na zwrocie — nie klasyfikacja reżimu
- Brak kosztów pożyczki, slippage, podatków
- Backtest ≠ wykonanie na żywo

### Sensowne kierunki rozwoju

1. **Klasyfikator reżimu** (risk-on / risk-off) zamiast regresji zwrotu
2. **Więcej aktywów** i portfel z ograniczeniami wag
3. **Live API** + paper trading
4. **Notebook Jupyter** do prezentacji interaktywnej

---

## 9. Zamknięcie i pytania

### Zdanie końcowe

> „Podsumowując: zbudowaliśmy **spójny pipeline** od danych po symulację ryzyka, z naciskiem na **metodologię** — brak wycieku, walk-forward, oddzielenie prognozy ML od decyzji inwestycyjnej. Model **nie udaje**, że przewiduje kierunek; pokazuje, gdzie **zarządzanie ekspozycją** może poprawić profil ryzyka w określonych reżimach. Kod jest modułowy, udokumentowany i dostępny na GitHubie.”

### Typowe pytania i krótkie odpowiedzi

| Pytanie | Odpowiedź |
|---------|-----------|
| Dlaczego log-zwroty? | Addytywność, lepsze właściwości statystyczne, standard w finansach ilościowych |
| Dlaczego GARCH? | Klastry zmienności; σ jest użyteczne dla ML i Monte Carlo |
| Dlaczego XGBoost, nie LSTM? | Mało danych dziennych; tabular features; interpretowalność |
| Czy można na tym zarabiać? | To badania; walk-forward nie gwarantuje przyszłości |
| Co jest najważniejszą metryką? | **Walk-forward** + porównanie drawdown i Sortino vs benchmark |

---

## Ściąga — najważniejsze liczby z Twojego uruchomienia

*Uzupełnij po każdym świeżym `python main.py` — wartości zależą od dat i rynku.*

| Aktywo | R² (średnie) | Dir. acc. | Walk-forward: zwrot strat. | Walk-forward: max DD strat. | Werdykt |
|--------|--------------|-----------|----------------------------|---------------------------|---------|
| BTC-USD | ~−0.30 | ~49.5% | *(wpisz z konsoli)* | *(wpisz)* | DEFENSYWNA PRZEWAGA |
| ETH-USD | ~−0.17 | ~48.6% | *(wpisz)* | *(wpisz)* | PEŁNA PRZEWAGA* |

\* ETH: pełna przewaga często wynika z ujemnego benchmarku w oknie testowym.

---

## Dodatek: plan slajdów (10–12 slajdów)

| Slajd | Treść |
|-------|--------|
| 1 | Tytuł, autor, link GitHub |
| 2 | Problem i hipoteza (kierunek vs zmienność) |
| 3 | Architektura 3 warstw (diagram) |
| 4 | Dane i log-zwroty |
| 5 | GARCH — wzór + wykres zmienności |
| 6 | Cechy ML + feature importance (screenshot) |
| 7 | XGBoost + TimeSeriesSplit |
| 8 | Strategia Long/Flat + walk-forward |
| 9 | Wyniki tabela BTC/ETH |
| 10 | Monte Carlo — wykresy z README |
| 11 | Ograniczenia i wnioski |
| 12 | Q&A |

---

*Dokument przeznaczony do wystąpienia ustnego. Można go drukować lub trzymać drugi monitor podczas prezentacji.*
