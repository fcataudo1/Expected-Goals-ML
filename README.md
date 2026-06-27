# ⚽ Predizione della probabilità di Goal da un tiro (Expected Goals – xG)

Progetto di **Machine Learning** sviluppato in ambito accademico per stimare la probabilità che un tiro nel calcio si concluda con un **Goal**, utilizzando il dataset pubblico **StatsBomb Open Data**.

Il problema è formulato come un task di **classificazione binaria supervisionata**, in cui il modello predice se un evento di tiro avrà come esito:

* **1 → Goal**
* **0 → No Goal**

L'intero progetto è stato realizzato implementando una pipeline di Machine Learning end-to-end, dalla preparazione dei dati fino alla valutazione e al confronto dei modelli.

---

## 📌 Scenario e Obiettivo

L'obiettivo del progetto è costruire un modello in grado di stimare gli **Expected Goals (xG)** di un tiro, apprendendo le caratteristiche che influenzano la probabilità di segnare.

### Task

* **Unità di analisi:** singolo evento di tiro (*shot*)
* **Tipo di problema:** classificazione binaria
* **Target:**

  * `1` → Goal
  * `0` → No Goal

### Principali sfide affrontate

* Forte sbilanciamento delle classi (goal rate ≈ 11%).
* Gestione di dati JSON con struttura annidata.
* Trattamento dei valori mancanti.
* Prevenzione del **data leakage**: la variabile `statsbomb_xg`, già presente nel dataset, è stata completamente esclusa dall'addestramento ed è stata utilizzata esclusivamente come benchmark finale.

---

## 📊 Dataset

Il progetto utilizza **StatsBomb Open Data**, una raccolta pubblica di eventi calcistici ampiamente impiegata nell'ambito della **Sport Analytics**.

Sono stati considerati esclusivamente gli eventi di tipo **Shot**, estratti cronologicamente dai file:

```text
events/{match_id}.json
```

---

## ⚙️ Pipeline di Machine Learning

La pipeline, sviluppata interamente in **Python** utilizzando **Pandas** e **Scikit-learn**, comprende le seguenti fasi:

1. Caricamento e parsing dei file JSON.
2. Pulizia dei dati e gestione dei valori mancanti.
3. Feature Engineering vettorializzato:

   * distanza dalla porta;
   * angolo di tiro.
4. Encoding delle variabili categoriche mediante `ColumnTransformer` e `OneHotEncoder`.
5. Suddivisione stratificata in training e test set.
6. Addestramento dei modelli tramite Pipeline di Scikit-learn.
7. Valutazione delle prestazioni e analisi grafica dei risultati.

---

## 🤖 Modelli e Risultati

Sono stati addestrati e confrontati tre algoritmi di classificazione.

Considerato il forte sbilanciamento delle classi, la valutazione è stata effettuata utilizzando metriche robuste:

* Precision
* Recall
* F1-Score
* ROC-AUC
* Average Precision

| Modello           |  ROC-AUC  | Osservazioni                                                                                                |
| ----------------- | :-------: | ----------------------------------------------------------------------------------------------------------- |
| **Naive Bayes**   |   0.755   | Modello di baseline con buona capacità discriminante ma recall limitata.                                    |
| **Decision Tree** |   0.604   | Prestazioni inferiori e tendenza all'overfitting.                                                           |
| **Random Forest** | **0.777** | Miglior modello ottenuto, con stime probabilistiche (Expected Goals) coerenti con il benchmark del dataset. |

---

## 🛠️ Tecnologie Utilizzate

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* StatsBomb Open Data

---

## 📂 Struttura del Progetto

```text
ProgettoML/
│
├── main.py
├── src/
│   ├── load_data.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── evaluate.py
│
├── data/
│   └── events/                  # File JSON di StatsBomb
│
├── documentazione/
│   ├── report.pdf               # Relazione di progetto
│   └── presentazione.pptx       # Slide della presentazione
│
├── plots/                       # Curve ROC, PR e Confusion Matrix
├── requirements.txt
└── README.md
```

---

## ▶️ Esecuzione

### 1. Inserire il dataset

Posizionare i file JSON del dataset StatsBomb nella cartella:

```text
data/events/
```

### 2. Installare le dipendenze

```bash
pip install -r requirements.txt
```

### 3. Avviare la pipeline

```bash
python main.py
```

Al termine dell'esecuzione verranno prodotti:

* i **Classification Report** dei tre modelli;
* le **Confusion Matrix**;
* le curve **ROC** e **Precision-Recall**;
* tutti i grafici di valutazione nella cartella `plots/`.

---

## 📚 Contesto Accademico

Progetto sviluppato nell'ambito del corso di **Machine Learning** presso l'**Università degli Studi di Salerno**.
