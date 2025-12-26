# main.py
from src.load_data import load_shots
from src.feature_engineering import build_features
from src.train_models import train_models
from src.evaluate import evaluate

def main():
    df = load_shots("data/data/events")
    print("Shots loaded:", df.shape)

    df = build_features(df)
    print("After feature engineering:", df.shape)
    print("Goal rate:", df["goal"].mean())

    trained = train_models(df)
    evaluate(trained)

if __name__ == "__main__":
    main()
