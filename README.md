# Hybrydowy Model Krypto (Crypto Hybrid Model)

Trójwarstwowy framework badawczo-analityczny dla walut Bitcoin (BTC) i Ethereum (ETH), który łączy klasyczną ekonometrię statystyczną (GARCH) z zaawansowanym uczeniem maszynowym (XGBoost) oraz probabilistycznymi symulacjami ryzyka (Monte Carlo). Framework służy do prognozowania dziennych zwrotów rynkowych oraz zarządzania kapitałem za pomocą strategii typu *Long/Flat* (w rynku lub w gotówce).

---

## Intuicja projektu

Przewidywanie, czy cena Bitcoina jutro wzrośnie, czy spadnie, jest niesamowicie trudne i przypomina rzut monetą – rynek jest pełen szumu. Istnieje jednak pewna prawidłowość: **ryzyko (zmienność) na rynku finansowym lubi się kumulować**. Jeśli dzisiaj na giełdzie dzieje się coś gwałtownego, istnieje ogromne prawdopodobieństwo, że jutro również zobaczymy potężne wahania.

Ten model nie próbuje rozpaczliwie odgadnąć przyszłej ceny. Jego zadaniem jest **zrozumieć reżim ryzyka**: uczy się, kiedy na rynku panuje sztorm, a kiedy spokój, i na tej podstawie decyduje, czy bezpieczniej jest inwestować (Long), czy przeczekać kryzys w gotówce (Flat).

---

## 🗺️ Architektura Systemu i Przepływ Danych

System został zaprojektowany w sposób całkowicie modułowy, izolując od siebie pobieranie danych, matematykę statystyczną, algorytmy sztucznej inteligencji oraz symulację przyszłości.

```text
[ WARSTWA 0: DANE ]
Pobieranie surowych notowań OHLCV z Yahoo Finance (yfinance) -> Obliczanie log-zwrotów
│
▼
[ WARSTWA 1: STATYSTYKA ]
Test stacjonarności ADF -> Dopasowanie modelu GARCH(1,1) -> Wygenerowanie cechy zmienności (σ)
│
▼
[ WARSTWA 2: UCZENIE MASZYNOWE ]
Inżynieria cech (Lags, RSI, SMA) -> Trening XGBoost -> Walidacja czasowa (TimeSeriesSplit)
│
▼
[ WARSTWA 3: ZARZĄDZANIE RYZYKIEM ]
Zwektoryzowany Ruch Browna (Monte Carlo) -> Kalkulacja VaR/CVaR -> Kryterium Kelly'ego / Sharpe'a
```

---

## 📈 Warstwa 1: Statystyka i Podwaliny Ekonometryczne

Zanim dane trafią do algorytmów sztucznej inteligencji, poddawane są rygorystycznej obróbce statystycznej. Pozwala to na dostarczenie modelowi ML cech o najwyższej wartości informacyjnej.

### 1. Log-zwroty (Log-Returns)

W analizie finansowej rzadko operuje się na surowych cenach. Zmian procentowych nie da się łatwo sumować, dlatego model oblicza logarytmiczne zwroty z cen zamknięcia według wzoru:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

Gdzie \(P_t\) to cena w dniu dzisiejszym, a \(P_{t-1}\) to cena z dnia poprzedniego.

- **Dlaczego to robimy?** Log-zwroty mają doskonałe właściwości matematyczne – przede wszystkim są addytywne w czasie. Oznacza to, że suma log-zwrotów z kilku kolejnych dni daje dokładnie log-zwrot z całego tego okresu.

### 2. Test Stacjonarności ADF (Augmented Dickey-Fuller)

Większość modeli statystycznych i uczenia maszynowego wymaga, aby dane wejściowe były stacjonarne (czyli by ich średnia i wariancja nie zmieniały się drastycznie w czasie). Test ADF sprawdza hipotezę zerową mówiącą o tym, że szereg czasowy posiada pierwiastek jednostkowy (jest niestacjonarny). Niski wskaźnik p-value pozwala odrzucić tę hipotezę i uznać zwroty za stacjonarne, co uzasadnia ich dalsze procesowanie.

### 3. Model GARCH(1,1) – Dynamiczna Analiza Ryzyka

Model GARCH (Generalized Autoregressive Conditional Heteroskedasticity) służy do prognozowania zmiennej w czasie wariancji (czyli dynamicznego ryzyka). Klasyczny model GARCH(1,1) opisuje warunkową wariancję \(\sigma_t^2\) równaniem:

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

- \(\omega\) – stały poziom wariancji (szum bazowy),
- \(\alpha \varepsilon_{t-1}^2\) – wpływ wczorajszego szoku cenowego (gwałtownej informacji z rynku),
- \(\beta \sigma_{t-1}^2\) – wpływ wczorajszego poziomu zmienności (efekt pamięci rynkowej).

Wygenerowana w ten sposób prognoza zmienności dziennej przekształcana jest na wartość roczną (annualizowaną) z uwzględnieniem faktu, że rynki kryptowalut działają bez przerwy (365 dni w roku):

$$
\sigma_{\mathrm{roczna}} = \sigma_{\mathrm{dziennie}} \times \sqrt{365}
$$

Wartość ta zasila model uczenia maszynowego jako kluczowa cecha opisująca bieżący poziom strachu/chciwości na rynku oraz stanowi parametr wejściowy do symulacji ryzyka.

---

## 🤖 Warstwa 2: Uczenie Maszynowe (Głębokie Wyjaśnienie XGBoost)

Warstwa uczenia maszynowego odpowiada za syntezę wszystkich danych rynkowych i próbę prognozy siły ruchu ceny na kolejny dzień.

### Sformułowanie problemu: Regresja zamiast Klasyfikacji

Projekt podchodzi do problemu w sposób ciągły. Nie klasyfikujemy ruchu jako proste „góra/dół” (0 lub 1). Model rozwiązuje zadanie regresji – uczy się przewidywać dokładną wartość liczbową (prognozowany log-zwrot następnego dnia). Dopiero w kolejnym kroku, na podstawie optymalizacji matematycznej pod kątem wskaźnika Sortino, dopasowywany jest próg decyzyjny aktywujący pozycję na rynku.

### Inżynieria Cech (Feature Engineering) – Co widzi model?

Algorytmy drzewiaste nie potrafią ekstrapolować danych poza zakres, który widziały podczas treningu. Podanie im surowej ceny Bitcoina sprawiłoby, że model stałby się bezużyteczny po przebiciu historycznych maksimów. Dlatego wszystkie cechy transformowane są do postaci relatywnej i stacjonarnej:

| Grupa cech | Opis mechanizmu | Rola w modelu |
|---|---|---|
| Opóźnienia (Lags 1-7d) | Historyczne log-zwroty z ostatnich 7 dni. | Badanie krótkoterminowej autokorelacji i pędu. |
| Stosunki SMA | Relacja bieżącej ceny do jej średnich kroczących (np. SMA 7 do SMA 21). | Identyfikacja trendu bez podawania nominalnej ceny. |
| RSI-14 | Wskaźnik siły relatywnej (Relative Strength Index). | Wykrywanie poziomów wykupienia i wyprzedania rynku. |
| Zmienność GARCH | Dynamiczna, warunkowa zmienność wyliczona w Warstwie 1. | Dostarczanie czystej informacji o aktualnym reżimie ryzyka. |
| Dynamika Wolumenu | Procentowe zmiany oraz stosunki wolumenu obrotu. | Potwierdzenie istotności ruchów cenowych przez kapitał. |

### Jak działa algorytm XGBoost?

XGBoost (Extreme Gradient Boosting) to jeden z najpotężniejszych algorytmów uczenia maszynowego dla danych tabelarycznych. Należy on do rodziny metod zespołowych (Ensemble Learning) i opiera się na koncepcji wzmacniania gradientowego (Gradient Boosting).

#### 🧠 Intuicja obrazowa: Sztafeta Ekspertów

Wyobraź sobie, że grupa ekspertów próbuje wspólnie rozwiązać trudne zadanie matematyczne:

1. **Pierwsze drzewo decyzyjne (Ekspert nr 1)** buduje bardzo prostą regułę, np.: „Jeśli RSI < 30, to jutrzejszy zwrot wyniesie +0.5%”. Popełnia przy tym ogromny błąd (tzw. residuum), ponieważ rynek jest skomplikowany.
2. **Drugie drzewo decyzyjne (Ekspert nr 2)** nie próbuje przewidzieć zwrotu od nowa. Jego jedynym zadaniem jest przewidzieć błąd popełniony przez Eksperta nr 1 i go skorygować.
3. **Trzecie drzewo (Ekspert nr 3)** uczy się przewidywać błąd pozostałej dwójki.

Każde kolejne drzewo jest celowo trenowane na porażkach swoich poprzedników. Końcowa prognoza to suma składowych wszystkich drzew, gdzie każde kolejne koryguje niedociągnięcia poprzednich.

```text
[Surowe Dane] ──> (Drzewo 1: Prognoza bazowa) ──> Obliczenie Błędu (Residuum)
│
▼
(Drzewo 2: Przewiduje Błąd Drzewa 1)
│
▼
(Drzewo 3: Przewiduje Błąd Drzewa 2)
│
▼
[Ostateczny Wynik] <── Suma ważona wszystkich korekt punktowych
```

### 🛠️ Dlaczego XGBoost, a nie sieci neuronowe (np. LSTM)?

- **Odporność na przeuczenie (Regularization):** XGBoost posiada wbudowaną penalizację za tworzenie zbyt skomplikowanych drzew (parametry L1 i L2). Zapobiega to dopasowywaniu się modelu do czystego szumu rynkowego.
- **Geometria podziału przestrzeni:** Drzewa decyzyjne świetnie radzą sobie z nieliniowymi zależnościami interakcji cech (np. „jeśli RSI > 70 ORAZ zmienność GARCH jest ekstremalnie wysoka, to wyjdź z rynku”).
- **Wielkość zbioru danych:** Przy dziennych danych historycznych dysponujemy kilkoma tysiącami próbek. Dla głębokich sieci neuronowych to zbyt mało, natomiast dla XGBoost jest to wolumen optymalny.

### Strażnicy Poprawności Metodologicznej: Chronologia Chroniona

W prognozowaniu finansowym najłatwiej o błąd zwany wyciekiem danych (Data Leakage). Jeśli model przypadkowo pozna chociażby ułamek informacji z przyszłości, wyniki testów będą genialne, a system na żywo straci kapitał. W projekcie zaimplementowano dwa mechanizmy obronne:

#### 1. TimeSeriesSplit

Klasyczna walidacja krzyżowa (K-Fold) losowo dzieli dane, co w szeregach czasowych łamie chronologię. Stosujemy podział kroczący z zachowaniem kierunku czasu:

```text
Fold 1: [ Trening: Rok 2021 ] ──> [ Test: Rok 2022 ]
Fold 2: [ Trening: Rok 2021-2022 ] ──> [ Test: Rok 2023 ]
Fold 3: [ Trening: Rok 2021-2023 ] ──> [ Test: Rok 2024 ]
```

#### 2. Kalibracja Walk-Forward

Model ML generuje surową prognozę zwrotu. Moduł `backtest.py` szuka optymalnego progu odcięcia (np. aktywuj pozycję Long tylko wtedy, gdy prognoza przekracza wyliczoną barierę), minimalizując ryzyko fałszywych sygnałów. Optymalizacja ta odbywa się wyłącznie na danych historycznych (in-sample) względem danego okna testowego, symulując rzeczywiste warunki handlu.

---

## 🎲 Warstwa 3: Zarządzanie Ryzykiem i Symulacje Monte Carlo

Ostatnia warstwa odpowiada za propagację niepewności w przód. Wykorzystując prognozy z poprzednich kroków, silnik uruchamia 10 000 niezależnych symulacji rozwoju ceny na najbliższe 30 dni.

### Geometryczny Ruch Browna (GBM)

Ścieżki cenowe generowane są za pomocą stochastycznego równania różniczkowego w postaci zdyskretyzowanej (krok dzienny):

$$
S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right]
$$

- \(S_0\) – ostatnia znana cena rzeczywista (punkt startowy),
- \(\mu\) – dryf rynkowy pobrany bezpośrednio z prognozy modelu XGBoost (roczny oczekiwany zwrot),
- \(\sigma\) – zmienność warunkowa pobrana z modelu GARCH(1,1),
- \(\Delta t\) – krok czasowy (\(1/365\)),
- \(Z\) – losowa zmienna o rozkładzie standardowym normalnym \(\mathcal{N}(0,1)\) odpowiedzialna za generowanie 10 000 wariantów.

### Miary Ryzyka Ogona (Tail Risk)

Z końcowego rozkładu symulowanych cen wyliczane są kluczowe wskaźniki dla zarządzania ryzykiem:

- **Value at Risk (VaR 95% / 99%):** Maksymalna oczekiwana strata, jakiej można się spodziewać w horyzoncie 30 dni z prawdopodobieństwem odpowiednio 95% lub 99%.
- **Conditional Value at Risk (CVaR / Expected Shortfall):** Średnia strata w najgorszych 5% lub 1% scenariuszy (pokazuje głębokość zapaści kapitału po przekroczeniu bariery VaR).

---

## 🖼️ Wykresy i Symulacje

W oparciu o parametry wejściowe, model generuje trajektorie cenowe oraz histogramy rozkładu ryzyka końcowego.

### Bitcoin (BTC-USD)

![Symulacja Monte Carlo BTC](montecarlo_BTC_USD.png)

### Ethereum (ETH-USD)

![Symulacja Monte Carlo ETH](montecarlo_ETH_USD.png)

---

## 🛠️ Szybki Start

### 1. Przygotowanie środowiska i instalacja

Projekt wymaga Pythona w wersji 3.10 lub nowszej.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

### 2. Wykonanie rurociągu badawczego

```powershell
# Pełny pipeline (konsola + zapis wykresów PNG do plików)
python main.py

# Pokazywanie okien wykresów matplotlib (wymaga ich zamykania w celu kontynuacji)
python main.py --show-plots
```

---

## 📊 Interpretacja Wyników (Poradnik akademicki)

Uruchomienie potoku wygeneruje rozbudowany raport tekstowy. Oto jak należy go interpretować:

- **R² < 0:** W finansach ilościowych na danych dziennych ujemne \(R^2\) jest rzeczą normalną. Oznacza to, że stosunek sygnału do szumu jest krytycznie niski, a model radzi sobie słabiej niż przewidywanie prostej średniej.
- **Trafność kierunkowa ~49%:** Rynek jest bliski błądzenia losowego. Algorytm zarabia nie na trafnym odgadywaniu każdego ruchu, ale na unikaniu dużych obsunięć kapitału (drawdown) poprzez ucieczkę w gotówkę w reżimach wysokiego ryzyka.

### Klasyfikacja Werdyktów Inwestycyjnych

- **PEŁNA PRZEWAGA:** Strategia handlowa pokonuje benchmark pasywny (Buy & Hold) zarówno pod kątem zwrotu, jak i metryk skorygowanych o ryzyko.
- **DEFENSYWNA PRZEWAGA:** Algorytm dostarcza mniejszych obsunięć kapitału (Max Drawdown) i ma lepsze wskaźniki Sharpe/Sortino, lecz wygenerował mniejszy całkowity zysk (profil: bezpieczniej, ale mniej zyskownie).
- **BRAK PRZEWAGI:** Wyniki strategii są gorsze od standardowego trzymania kryptowaluty.

---

## 📢 Skrypt Prezentacyjny (Wskazówki na wystąpienie)

Jeśli prezentujesz ten projekt przed zespołem technologicznym lub wykładowcą, oprzyj strukturę o poniższe punkty:

1. **Problem i Hipoteza:** Dzienne zwroty kryptowalut są bliskie białemu szumowi (nieprzewidywalne kierunkowo), ale ich zmienność wykazuje klastrowanie (ryzyko da się modelować).
2. **Rygor Metodologiczny:** Zapobiegliśmy wyciekowi danych poprzez zaimplementowanie TimeSeriesSplit oraz zagnieżdżoną kalibrację progów wyłącznie in-sample.
3. **Wnioski z wag cech:** Model XGBoost najwyższe znaczenie przypisuje wskaźnikom zmienności (GARCH oraz Rolling Volatility), co dowodzi, że algorytm prawidłowo nauczył się reagować na reżimy ryzyka zamiast na surowe poziomy cenowe.

---

## ⚠️ Zastrzeżenie prawne (Disclaimer)

Zbudowany algorytm oraz dostarczane przez niego wyniki mają charakter wyłącznie badawczy, statystyczny i edukacyjny. W żadnym wypadku projekt nie stanowi doradztwa inwestycyjnego ani zachęty do obrotu prawdziwymi środkami finansowymi.
