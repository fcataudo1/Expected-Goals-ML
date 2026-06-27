# ⚽ Predizione Goal da un tiro (Expected Goals – xG)

Questo progetto ha l’obiettivo di realizzare una **soluzione di Machine Learning** per la predizione dell’esito di un tiro nel calcio, utilizzando il dataset pubblico **StatsBomb Open Data**.
Il problema è formulato come un task di **classificazione binaria supervisionata**, in cui si stima la probabilità che un evento di tiro si concluda con un **Goal** (1) oppure **No Goal** (0).

Il progetto è stato sviluppato applicando le tecniche e gli approcci di Machine Learning studiati in ambito accademico, implementando una pipeline end-to-end robusta e documentata.

---

## 📌 Scenario e Task
- **Unità di analisi**: Singolo evento di tiro (*shot*)
- **Target**:
  - `1` → Goal
  - `0` → No Goal
- **Tipo di problema**: Classificazione binaria
- **Principali sfide affrontate**:
  - Forte sbilanciamento delle classi (goal rate ~11%).
  - Gestione di valori mancanti e struttura annidata dei dati JSON.
  - Prevenzione assoluta del *data leakage* (il valore `statsbomb_xg` fornito dal dataset è stato isolato e utilizzato esclusivamente come benchmark di confronto a posteriori).

---

## 📊 Dataset
Il dataset utilizzato è lo **StatsBomb Open Data**, ampiamente impiegato per la sport analytics.
Per il progetto vengono considerati esclusivamente gli eventi di tipo **Shot**, estratti cronologicamente dai file `events/{match_id}.json`.

---

## ⚙️ Pipeline di Machine Learning
La pipeline implementata, sviluppata interamente in Python (Pandas, Scikit-Learn), comprende i seguenti step:

1. **Caricamento e parsing sicuro dei dati JSON**
2. **Pulizia e gestione dei valori mancanti**
3. **Feature Engineering Vettorializzato**: Calcolo di *Distanza dalla porta* e *Angolo di tiro* a partire dalle coordinate spaziali grezze.
4. **Encoding delle feature categoriche** (via `ColumnTransformer` e `OneHotEncoder`)
5. **Suddivisione train/test stratificata** per preservare la reale distribuzione dei goal.
6. **Addestramento dei modelli in Pipeline**
7. **Valutazione delle prestazioni e analisi grafica**

---

## 🤖 Modelli e Risultati
Sono stati addestrati e confrontati tre modelli di classificazione. Considerato il forte sbilanciamento, la valutazione si è basata su metriche robuste (Precision, Recall, F1-Score, ROC-AUC, Average Precision).

| Modello | ROC-AUC | Note |
| :--- | :---: | :--- |
| **Naive Bayes** | 0.755 | Modello di baseline, buona separazione generale ma alto tasso di falsi negativi. |
| **Decision Tree** | 0.604 | Scarsa capacità di generalizzazione, tendenza all'overfitting sui falsi positivi. |
| **Random Forest** | **0.777** | **Modello ottimale.** Drastica riduzione dei falsi positivi e stime probabilistiche (xG) coerenti con il benchmark di riferimento. |

---

## 📂 Struttura del progetto

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
    │   └── events/           (Inserire qui i file JSON di StatsBomb)
    │
    ├── plots/                (Generata in automatico: Curve ROC, PR e Confusion Matrix)
    ├── report.pdf            Relazione di progetto e discussione dei risultati
    ├── requirements.txt      Dipendenze del progetto
    └── README.md

---

## ▶️ Esecuzione
1. Posizionare i file `events/*.json` del dataset StatsBomb nella cartella:
   `data/events/`
   
2. Installare le dipendenze:
   `pip install -r requirements.txt`
   
3. Eseguire la pipeline completa:
   `python main.py`

L’esecuzione produrrà a schermo i *Classification Report* per i tre modelli e salverà nella cartella `/plots` tutti i grafici di valutazione comparativa e le singole matrici di confusione.

3.Eseguire il progetto:
python main.py

L’esecuzione produrrà:
- metriche di valutazione a terminale
- grafici (ROC, Precision–Recall, Confusion Matrix) nella cartella plots/
