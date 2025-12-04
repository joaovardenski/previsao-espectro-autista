from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_models(X, y):
    rf = RandomForestClassifier(random_state=42)
    xt = ExtraTreesClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42, eval_metric="logloss")

    param_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    param_xgb = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1],
    }

    rf_best = RandomizedSearchCV(rf, param_rf, n_iter=8, cv=5, scoring="roc_auc")
    xgb_best = RandomizedSearchCV(xgb, param_xgb, n_iter=8, cv=5, scoring="roc_auc")

    rf_best.fit(X, y)
    xgb_best.fit(X, y)
    xt.fit(X, y)

    estimators = [
        ("rf", rf_best.best_estimator_),
        ("xgb", xgb_best.best_estimator_),
        ("xt", xt)
    ]

    meta = LogisticRegression(max_iter=1000, solver="liblinear")

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=5
    )

    return stacking, rf_best.best_estimator_, xt, xgb_best.best_estimator_
