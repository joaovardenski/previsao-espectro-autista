# 🧠 Autism Spectrum Prediction

Previsão de traços do espectro autista usando Machine Learning

Este projeto utiliza técnicas de Ciência de Dados e Machine Learning
para prever indicadores associados ao espectro autista, explorando um
pipeline completo que envolve pré-processamento, balanceamento,
treinamento e comparação de modelos.

## 📌 Objetivos do Projeto

-   Criar um pipeline totalmente estruturado para predição.
-   Aplicar boas práticas de ML: normalização, encoding, separação
    treino/teste, balanceamento etc.
-   Testar modelos como XGBoost, Random Forest e Regressão Logística.
-   Avaliar desempenho usando métricas e visualizações.
-   Organizar o código em módulos reutilizáveis.

## 🗂 Estrutura do Projeto

    ├── src
    │   ├── preprocessing.py     # Funções de limpeza, encoding, normalização e SMOTE
    │   ├── models.py            # Definição e inicialização de modelos de ML
    │   ├── train.py             # Pipeline de treinamento e avaliação
    ├── main.py                  # Script principal de execução
    ├── requirements.txt         # Dependências do projeto
    └── README.md

## ⚙️ Tecnologias Utilizadas

### 🐍 Linguagem

Python 3.12.3

### 📚 Bibliotecas

-   Pandas\
-   NumPy\
-   Scikit-Learn\
-   XGBoost\
-   Imbalanced-learn (SMOTE)\
-   Matplotlib / Seaborn

## 🚀 Como executar o projeto

### 1. Clone o repositório

    git clone https://github.com/joaovardenski/previsao-espectro-autista.git
    cd previsao-espectro-autista

### 2. Crie o ambiente virtual

    python -m venv venv

Ative o ambiente:

**Windows:**

    venv\Scripts\activate

**Linux/Mac:**

    source venv/bin/activate

### 3. Instale as dependências

    pip install -r requirements.txt

### 4. Execute o projeto

    python main.py

## 📊 Modelos Avaliados

Os seguintes algoritmos foram utilizados para comparação:

-   XGBoost\
-   Random Forest\
-   Logistic Regression\
-   Decision Tree\
-   KNN

As métricas avaliadas incluem:

-   Acurácia\
-   Precision\
-   Recall\
-   F1-score\
-   Matriz de confusão

## 🧪 Sobre o Dataset

O dataset contém variáveis relacionadas a padrões comportamentais e
características clínicas.\
O projeto aplica pré-processamento para normalizar, corrigir e preparar
os dados.\
Foi utilizado SMOTE para lidar com desbalanceamento.

## 📈 Resultados

Os resultados incluem:

-   Comparação das métricas entre modelos\
-   Identificação do modelo com melhor desempenho\
-   Insights extraídos da análise exploratória

## 🤝 Contribuições

Sinta-se livre para abrir issues, enviar sugestões ou fazer pull
requests.\
Melhorias no pipeline e modelos são sempre bem-vindas.

## 📄 Licença

Este projeto está sob a licença MIT --- utilize e modifique livremente.
