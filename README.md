# Predizione Goal da un tiro (Expected Goals – xG)

Questo progetto ha l’obiettivo di realizzare una **soluzione di Machine Learning** per la predizione dell’esito di un tiro nel calcio, utilizzando il dataset **StatsBomb Open Data**.  
Il problema è formulato come un task di **classificazione binaria supervisionata**, in cui si prevede se un evento di tiro si concluderà con un **Goal** oppure **No Goal**.

Il progetto è stato sviluppato applicando esclusivamente le tecniche e gli approcci di Machine Learning studiati durante il corso, seguendo un’intera pipeline end-to-end.

---

## 📌 Scenario e Task
- **Unità di analisi**: evento di tiro (*shot*)
- **Target**:
  - `1` → Goal
  - `0` → No Goal
- **Tipo di problema**: classificazione binaria
- **Principali difficoltà**:
  - forte sbilanciamento delle classi
  - presenza di valori mancanti
  - struttura annidata dei dati
  - relazioni non lineari tra le feature

---

## 📊 Dataset
Il dataset utilizzato è **StatsBomb Open Data**, che fornisce eventi calcistici in formato JSON.  
Per il progetto vengono considerati esclusivamente gli eventi di tipo **Shot**, estratti dai file `events/{match_id}.json`.

---

## ⚙️ Pipeline di Machine Learning
La pipeline implementata comprende i seguenti step:

1. **Caricamento e parsing dei dati**
2. **Pulizia e gestione dei valori mancanti**
3. **Feature engineering**:
   - distanza dalla porta
   - angolo di tiro
4. **Encoding delle feature categoriche**
5. **Suddivisione train/test stratificata**
6. **Addestramento dei modelli**
7. **Valutazione delle prestazioni**
8. **Analisi degli errori**

---

## 🤖 Modelli utilizzati
Sono stati addestrati e confrontati i seguenti modelli, studiati durante il corso:

- **Naive Bayes** (baseline)
- **Decision Tree**
- **Random Forest**

La **Random Forest** risulta il modello con le migliori prestazioni complessive.

---

## 📈 Metriche di valutazione
Considerata la natura sbilanciata del problema, sono state utilizzate metriche tradizionali ma informative:

- Accuracy (con discussione critica)
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Precision–Recall Curve

---

## 📂 Struttura del progetto
ProgettoML/
│
├── main.py
├── src/
│ ├── load_data.py
│ ├── feature_engineering.py
│ ├── train_models.py
│ └── evaluate.py
│
├── data/
│ └── events/ # file events StatsBomb
│
├── plots/ # grafici generati
├── report.pdf
├── requirements.txt
└── README.md


---

## ▶️ Esecuzione
1. Posizionare i file `events/*.json` del dataset StatsBomb nella cartella:

data/events/
2. Installare le dipendenze:
```bash
pip install -r requirements.txt
```

3.Eseguire il progetto:
python main.py

L’esecuzione produrrà:
- metriche di valutazione a terminale
- grafici (ROC, Precision–Recall, Confusion Matrix) nella cartella plots/