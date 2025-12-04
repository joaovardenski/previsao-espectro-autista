import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd


def plot_target_distribution(df):
    plt.figure(figsize=(6,5))
    ax = sns.countplot(x="Class/ASD", data=df)
    plt.title("Distribuição da Classe Alvo")
    plt.show()


def plot_correlation(X_res, columns):
    corr = pd.DataFrame(X_res, columns=columns).corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlação das Features")
    plt.show()


def plot_confusion_matrix(model, X_val, y_val):
    ConfusionMatrixDisplay.from_estimator(model, X_val, y_val, cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.show()
