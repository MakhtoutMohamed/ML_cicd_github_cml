"""
Atelier 3 — CI/CD pour le Machine Learning (GitHub Actions + CML)
Projet pédagogique : churn-cml

Ce script :
- lit dataset.csv (dans le repo),
- prépare les données (imputation, encodage, scaling),
- entraîne 3 modèles RandomForest (baseline / class_weight / SMOTE),
- génère :
    - metrics.txt (résumé lisible),
    - conf_matrix.png (matrice de confusion combinée).
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # important en CI (pas d'affichage)
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from imblearn.over_sampling import SMOTE

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Sélectionne des colonnes d'un DataFrame Pandas (mini helper pour Pipeline sklearn)."""
    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Renvoie un DataFrame (pour garder le type/colonnes) ; sklearn gère ensuite
        return X[self.columns]


# --------------------- Data Preparation ---------------------------- #

TRAIN_PATH = os.path.join(os.getcwd(), "dataset.csv")
df = pd.read_csv(TRAIN_PATH)

# Drop first 3 features
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Filter using Age threshold
df.drop(index=df[df["Age"] > 80].index.tolist(), axis=0, inplace=True)

# Features / target
X = df.drop(columns=["Exited"])
y = df["Exited"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=45,
    stratify=y,
)

# --------------------- Data Processing ---------------------------- #

num_cols = ["Age", "CreditScore", "Balance", "EstimatedSalary"]
categ_cols = ["Gender", "Geography"]
ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))

num_pipeline = Pipeline(steps=[
    ("selector", DataFrameSelector(num_cols)),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categ_pipeline = Pipeline(steps=[
    ("selector", DataFrameSelector(categ_cols)),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", sparse_output=False)),
])

ready_pipeline = Pipeline(steps=[
    ("selector", DataFrameSelector(ready_cols)),
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

all_pipeline = FeatureUnion(transformer_list=[
    ("numerical", num_pipeline),
    ("categorical", categ_pipeline),
    ("ready", ready_pipeline),
])

X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)

# --------------------- Imbalance handling ---------------------------- #

# class weights (simple heuristic)
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)  # normalize

dict_weights = {i: vals_count[i] for i in range(2)}

# SMOTE oversampling
over = SMOTE(sampling_strategy=0.7, random_state=45)
X_train_resampled, y_train_resampled = over.fit_resample(X_train_final, y_train)

# --------------------- Modeling / Metrics ---------------------------- #

# Clear metrics.txt file at the beginning
with open("metrics.txt", "w", encoding="utf-8") as f:
    pass


def train_model(X_tr, y_tr, plot_name="", class_weight=None):
    """Train RandomForest and write metrics + plot confusion matrix."""
    global clf_name

    clf = RandomForestClassifier(
        n_estimators=300,     # réduit pour CI (plus rapide)
        max_depth=10,
        random_state=45,
        class_weight=class_weight,
        n_jobs=-1,
    )

    clf.fit(X_tr, y_tr)

    y_pred_train = clf.predict(X_tr)
    y_pred_test = clf.predict(X_test_final)

    score_train = f1_score(y_tr, y_pred_train)
    score_test = f1_score(y_test, y_pred_test)

    clf_name = clf.__class__.__name__

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred_test),
        annot=True,
        cbar=False,
        fmt=".2f",
        cmap="Blues",
    )
    plt.title(plot_name)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

    plt.savefig(f"{plot_name}.png", bbox_inches="tight", dpi=300)
    plt.close()

    with open("metrics.txt", "a", encoding="utf-8") as f:
        f.write(f"{clf_name} {plot_name}\n")
        f.write(f"F1-score of Training is: {score_train * 100:.2f} %\n")
        f.write(f"F1-Score of Validation is: {score_test * 100:.2f} %\n")
        f.write("----" * 10 + "\n")


# 1) Baseline
train_model(X_tr=X_train_final, y_tr=y_train, plot_name="without-imbalance", class_weight=None)

# 2) With class_weight
train_model(X_tr=X_train_final, y_tr=y_train, plot_name="with-class-weights", class_weight=dict_weights)

# 3) With SMOTE
train_model(X_tr=X_train_resampled, y_tr=y_train_resampled, plot_name="with-SMOTE", class_weight=None)

# Combine all confusion matrices into one image
confusion_matrix_paths = ["./without-imbalance.png", "./with-class-weights.png", "./with-SMOTE.png"]

plt.figure(figsize=(15, 5))
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis("off")

plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("conf_matrix.png", bbox_inches="tight", dpi=300)
plt.close()

# Optionnel: nettoyage des 3 PNG intermédiaires (décommenter si vous voulez)
# for p in confusion_matrix_paths:
#     os.remove(p)
