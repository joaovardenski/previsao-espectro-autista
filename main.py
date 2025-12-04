import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing import preprocess_data
from src.visualization import (
    plot_target_distribution,
    plot_correlation,
    plot_confusion_matrix
)
from src.models import train_models


def main():
    print("Carregando dataset...")
    df = pd.read_csv("data/train.csv")

    plot_target_distribution(df)

    print("Pré-processando...")
    X_res, y_res, columns = preprocess_data(df)

    plot_correlation(X_res, columns)

    print("Treinando modelos...")
    stacking, rf, xt, xgb = train_models(X_res, y_res)

    X_train, X_val, y_train, y_val = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_val)

    print("\nACCURACY:", accuracy_score(y_val, y_pred))
    print("\nREPORT:\n", classification_report(y_val, y_pred))

    plot_confusion_matrix(stacking, X_val, y_val)


if __name__ == "__main__":
    main()
