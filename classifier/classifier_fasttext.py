import csv
import os
from collections import Counter

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, recall_score, confusion_matrix, f1_score, roc_auc_score, \
    RocCurveDisplay, roc_curve
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random

from tqdm import tqdm
from xgboost import XGBClassifier

def plot_prediction_distribution(test_data, y_proba, threshold=0.5):
    """
    Affiche une droite avec la répartition des probabilités prédites (classe 1) pour chaque id_cas.
    Points rouges = vraie classe 1, points bleus = vraie classe 0
    """
    df = pd.DataFrame({
        "id_cas": [entry["id_cas"] for entry in test_data],
        "true_label": [entry["target"] for entry in test_data],
        "proba_1": y_proba
    })

    df["color"] = df["true_label"].map({0: "blue", 1: "red"})
    df["y"] = 0

    df = df.sort_values(by="proba_1").reset_index(drop=True)

    plt.figure(figsize=(14, 2.5))
    plt.scatter(df["proba_1"], df["y"], c=df["color"], alpha=0.7)

    plt.axvline(threshold, color='gray', linestyle='--', label=f"Seuil {threshold}")

    # Décorations
    plt.xlabel("Classe réelle 1 : rouge | Classe réelle 0 : bleu")
    plt.title("Probabilité prédite (par id_cas)")
    plt.yticks([])  # Pas besoin d'axe vertical
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_and_split_data(json_file, seed, test_ratio=0.15):
    random.seed(seed)
    np.random.seed(seed)

    with open(json_file) as f:
        data = json.load(f)

    data_0 = [entry for entry in data if entry["target"] == 0]
    data_1 = [entry for entry in data if entry["target"] == 1]

    # Taille du test set (même nombre pour chaque classe)
    n_test = min(int(len(data_0) * test_ratio), int(len(data_1) * test_ratio))

    test_0 = random.sample(data_0, n_test)
    test_1 = random.sample(data_1, n_test)

    test_data = test_0 + test_1

    # Création du train set avec les données restantes
    test_ids = {entry["id_cas"] for entry in test_data}
    train_0 = [entry for entry in data_0 if entry["id_cas"] not in test_ids]
    train_1 = [entry for entry in data_1 if entry["id_cas"] not in test_ids]

    # Rééquilibrage du train set
    n_train = min(len(train_0), len(train_1))
    train_0 = random.sample(train_0, n_train)
    train_1 = random.sample(train_1, n_train)

    train_data = train_0 + train_1
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data

def prepare_arrays(data):
    X = np.array([entry["embedding"] for entry in data])
    y = np.array([entry["target"] for entry in data])
    return X, y


def train_model(json_file, model_name, seed, threshold = 0.5):
    # Chargement et équilibrage
    train_data, test_data = load_and_split_data(json_file, seed)

    X_train, y_train = prepare_arrays(train_data)
    X_test, Y_test = prepare_arrays(test_data)

    # Choix du modèle
    if model_name == "SVM":
        model = SVC(kernel='linear',
                    probability=True,
                    random_state=seed)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=seed,
                                   max_iter=1000,
                                   class_weight="balanced",
                                   solver="liblinear")
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100,
                                       max_depth=100,
                                       random_state=seed)
    elif model_name == "XGBoost":
        model = XGBClassifier(booster="gbtree",
                              device = "cuda",
                              learning_rate = 0.05,
                              max_depth = 10,
                              eval_metric='logloss',
                              random_state=seed)

    model.fit(X_train, y_train)

    Y_proba_all = model.predict_proba(X_test)
    Y_proba = Y_proba_all[:, 1]
    Y_pred = (Y_proba >= threshold).astype(int)
    y_proba_1 = [proba[1] for proba in Y_proba_all]
    # ---------- Evaluation ----------------------

    score = recall_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_proba)
    fpr_test, tpr_test, _ = roc_curve(Y_test, Y_proba)


    # ----- TOUS LES PLOTS ---------

    plot_prediction_distribution(test_data, y_proba_1, threshold=0.5)


    cm = confusion_matrix(Y_test, Y_pred)
    classes = ['Non STEMI', 'STEMI']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=classes, yticklabels=classes, ax=axes[0], annot_kws={"size": 14}, vmin=0)
    axes[0].set_title(f"Matrice de confusion ({model_name})", fontsize=14)
    axes[0].set_xlabel("Prédiction", fontsize=12)
    axes[0].set_ylabel("Vérité terrain", fontsize=12)

    text_before = (f"Sensibilité (TP / (TP + FN)) : {score:.2f}"
                   f"\nSpécificité (TN / (TN + FP)) : {specificity:.2f}"
                    f"\nPrécision (TP / (TP + FP)) : {precision:.2f}"
                   f"\nF1 Score : {f1:.2f}")
    axes[0].text(0.5, -0.25, text_before, fontsize=12, ha='center', va='top', transform=axes[0].transAxes)

    axes[1].plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classifieur aléatoire')
    axes[1].set_title(f"Courbe ROC ({model_name})", fontsize=14)
    axes[1].set_xlabel("Taux de Faux Positifs", fontsize=12)
    axes[1].set_ylabel("Taux de Vrais Positifs", fontsize=12)
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    plt.show()

    # --------------- ENREGISTREMNT DES RESULTATS -------------------------
    output_dir = None
    if model_name == "SVM":
        output_dir = "SVM"
    elif model_name == "Logistic Regression":
        output_dir = "LR"
    elif model_name == "Random Forest":
        output_dir = "RF"
    elif model_name == "XGBoost":
        output_dir = "XGBoost"

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig()

    return int(score*100), int(specificity*100), int(precision*100), int(f1*100), int(roc_auc*100)


if __name__ == "__main__":

    classifiers = ["Logistic Regression", "Random Forest", "SVM", "XGBoost"] # SVM : plus prometteur ?
    data_file_fasstext_new = 'json_fasttext/new/json_sansmetadata.json'


    csv_file = "result_tfidf.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "seed", "sensibility", "specificity", "precision", "f1_score", "auc_roc"])

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        for classifier in classifiers:
            for i in range (1):
                seed = random.randint(1,10000)
                recall, specificity, precision, f1, roc_auc = train_model(data_file_fasstext_new, classifier, seed)
                writer.writerow([classifier, seed, recall, specificity, precision, f1, roc_auc])

