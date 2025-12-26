# src/train_models.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def _make_ohe_dense():
    # Compatibilità sklearn: sparse_output (nuovo) vs sparse (vecchio)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_models(df):
    """
    Ritorna un dict:
      name -> (pipeline, X_test, y_test, statsbomb_xg_test)
    """
    if "statsbomb_xg" not in df.columns:
        raise ValueError("Colonna 'statsbomb_xg' mancante. Aggiungila in load_data.py")

    # Feature per ML (ESCLUDIAMO statsbomb_xg per evitare leakage)
    X = df.drop(columns=["goal", "statsbomb_xg"])
    y = df["goal"]

    categorical = X.select_dtypes(include="object").columns.tolist()
    numerical = X.select_dtypes(exclude="object").columns.tolist()

    # Preprocessor per DT/RF (sparse ok)
    preprocessor_sparse = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ])

    # Preprocessor per NB (DEVE essere dense)
    preprocessor_dense = ColumnTransformer([
        ("cat", _make_ohe_dense(), categorical),
        ("num", "passthrough", numerical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # StatsBomb xG SOLO per confronto (allineato con gli stessi indici del test)
    statsbomb_xg_test = df.loc[X_test.index, "statsbomb_xg"]

    trained = {}

    # Naive Bayes (dense)
    nb_pipe = Pipeline([
        ("prep", preprocessor_dense),
        ("model", GaussianNB())
    ])
    nb_pipe.fit(X_train, y_train)
    trained["NaiveBayes"] = (nb_pipe, X_test, y_test, statsbomb_xg_test)

    # Decision Tree (sparse ok)
    dt_pipe = Pipeline([
        ("prep", preprocessor_sparse),
        ("model", DecisionTreeClassifier(random_state=42))
    ])
    dt_pipe.fit(X_train, y_train)
    trained["DecisionTree"] = (dt_pipe, X_test, y_test, statsbomb_xg_test)

    # Random Forest (sparse ok)
    rf_pipe = Pipeline([
        ("prep", preprocessor_sparse),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    rf_pipe.fit(X_train, y_train)
    trained["RandomForest"] = (rf_pipe, X_test, y_test, statsbomb_xg_test)

    return trained
