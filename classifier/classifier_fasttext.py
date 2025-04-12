import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, recall_score, confusion_matrix, f1_score, roc_auc_score, \
    RocCurveDisplay, roc_curve
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
from collections import Counter


def balance_data(data, method="undersample", seed=42):
    random.seed(seed)

    data_0 = [entry for entry in data if entry["target"] == 0]
    data_1 = [entry for entry in data if entry["target"] == 1]

    if method == "undersample":
        n = min(len(data_0), len(data_1))
        data_0 = random.sample(data_0, n)
        data_1 = random.sample(data_1, n)
    elif method == "oversample":
        n = max(len(data_0), len(data_1))
        if len(data_0) < n:
            data_0 = data_0 + random.choices(data_0, k=n - len(data_0))
        if len(data_1) < n:
            data_1 = data_1 + random.choices(data_1, k=n - len(data_1))
    else:
        raise ValueError("Méthode inconnue : utiliser 'oversample' ou 'undersample'")

    balanced = data_0 + data_1
    random.shuffle(balanced)
    return balanced


def classifier_training(json_file, model_name, balance_method='oversample', seed=42):
    random.seed(seed)
    np.random.seed(seed)

    with open(json_file) as f:
        data = json.load(f)

    # Séparation pour test équilibré
    data_0 = [entry for entry in data if entry["target"] == 0]
    data_1 = [entry for entry in data if entry["target"] == 1]
    n_test = min(len(data_0), len(data_1), int(0.15 * len(data)))

    test_0 = random.sample(data_0, n_test)
    test_1 = random.sample(data_1, n_test)

    test_data = test_0 + test_1
    train_data = [entry for entry in data if entry not in test_data]

    # 🎯 équilibrage du training set
    train_data = balance_data(train_data, method=balance_method, seed=seed)

    # Conversion en matrices
    X_train = np.array([entry["embedding"] for entry in train_data])
    Y_train = np.array([entry["target"] for entry in train_data])
    X_test = np.array([entry["embedding"] for entry in test_data])
    Y_test = np.array([entry["target"] for entry in test_data])

    print("Répartition train :", Counter(Y_train))
    print("Répartition test  :", Counter(Y_test))

    # Modèle
    if model_name == "SVM":
        model = SVC(kernel='linear', probability=True, random_state=seed)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=seed, max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=seed)
    else:
        raise ValueError("Modèle non reconnu")

    # Entraînement
    model.fit(X_train, Y_train)


    Y_pred = model.predict(X_test)
    Y_proba_all = model.predict_proba(X_test)
    Y_proba = Y_proba_all[:, 1]

    # ---------- Evaluation ----------------------

    score = recall_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_proba)

    print(f"Sensibilité : {score:.4f}")
    print(f"\nSpecificite (Taux Vrais Négatifs): {specificity:.4f}")
    print(f"\nDétail TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"\nROC AUC : {roc_auc:.4f}")
    print(f"\nF1 score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)


    # ----- Matrice de confusion ---------
    cm = confusion_matrix(Y_test, Y_pred)
    cmap = plt.cm.Oranges
    classes = ['Non STEMI', 'STEMI']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.yticks(rotation=0, va="center")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    fpr_test, tpr_test, _ = roc_curve(Y_test, Y_proba)
    plt.plot(fpr_test, tpr_test, color='orange', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classifieur aléatoire')
    plt.title("ROC Curve (SVM)")
    plt.grid(True)
    plt.show()


    return X_test, Y, model


def all_probabilities(json_file, svm_model, X_test):
    output_dir = "SVM"
    os.makedirs(output_dir, exist_ok=True)

    with open(json_file) as f:
        data = json.load(f)

    # --- Prédire les probabilités pour TOUTES les données ---
    all_probabilities = svm_model.predict_proba(X)

    # --- Créer la liste de résultats avec id, batch et probabilités ---
    results_with_probabilities = []
    for i, original_entry in enumerate(data):
        prob_class_0 = all_probabilities[i, 0]
        prob_class_1 = all_probabilities[i, 1]

        result_entry = {
            "id_cas": original_entry["id_cas"],
            "probability_0": prob_class_0,
            "probability_1": prob_class_1
        }
        output_filename_csv = "SVM_probabilites_fasttext.csv"


        results_with_probabilities.append(result_entry)


    csv_fieldnames = ["id_cas", "probability_0", "probability_1"]

    output_path = os.path.join(output_dir, output_filename_csv)

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(results_with_probabilities)

    print(f"Résultats avec probabilités enregistrés avec succès dans '{output_filename_csv}'")

if __name__ == "__main__":
    data_file_fasstext = 'json_fasttext/fasttext _age_n_sexe_padded.json'
    data_file_metadata = 'json_camembert/camembert_sans_metadata.json'
    # LR : "Logistic Regression"
    # RF : "Random Forest"
    # SVM : "SVM"
    model_name = "SVM"
    X, Y, svm_model = classifier_training(data_file_fasstext, model_name)
    all_probabilities(data_file_fasstext, svm_model, X)