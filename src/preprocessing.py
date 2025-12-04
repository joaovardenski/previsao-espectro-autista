import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def preprocess_data(df):
    cols_drop = [
        "ID", "age", "age_desc", "relation", "used_app_before",
        "contry_of_res", "ethnicity", "jaundice", "gender"
    ]
    df = df.drop(columns=cols_drop)

    label_cols = df.select_dtypes(include="object").columns
    le_dict = {}

    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df.drop(columns=["Class/ASD"])
    y = df["Class/ASD"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    return X_res, y_res, X.columns
