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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
from xgboost import XGBClassifier

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

def save_predictions_to_csv(test_data, y_proba_all, y_pred, filename="predictions_test.csv"):
    records = []
    for entry, proba, pred in zip(test_data, y_proba_all, y_pred):
        records.append({
            "id_cas": entry["id_cas"],
            "target": entry["target"],
            "proba_0": round(proba[0], 5),
            "proba_1": round(proba[1], 5),
            "prediction": int(pred)
        })
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)


def classifier_training(json_file, model_name, balance_method='undersample', seed=30, threshold=0.5):
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

    # équilibrage du training set
    train_data = balance_data(train_data, method=balance_method, seed=seed)

    # Conversion en matrices
    X_train = np.array([entry["embedding"] for entry in train_data])
    Y_train = np.array([entry["target"] for entry in train_data])
    X_test = np.array([entry["embedding"] for entry in test_data])
    Y_test = np.array([entry["target"] for entry in test_data])

    # Modèle
    if model_name == "SVM":
        model = SVC(kernel='linear', probability=True, random_state=seed)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced", solver="liblinear")
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=seed)
    elif model_name == "XGBoost":
        model = XGBClassifier(dart_normalized_type="forest", learning_rate=0.05, max_iterations=50, max_depth=10,
                          use_label_encoder=False, eval_metric='logloss', random_state=seed)

    else:
        raise ValueError("Modèle non reconnu")

    # Entraînement
    model.fit(X_train, Y_train)


    Y_proba_all = model.predict_proba(X_test)
    Y_proba = Y_proba_all[:, 1]
    Y_pred = (Y_proba >= threshold).astype(int)

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
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 14}, vmin = 0)

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

    save_predictions_to_csv(test_data, Y_proba_all, Y_pred, filename="predictions_test.csv")



if __name__ == "__main__":
    data_file_fasstext_old = 'json_fasttext/old/fasttext_sans_metadata_padded.json'
    data_file_fasstext_new = 'json_fasttext/new/json_sans_metadonnee.json'

    # LR : "Logistic Regression"
    # RF : "Random Forest"
    # SVM : "SVM"
    #
    model_name = "XGBoost"
    classifier_training(data_file_fasstext_old, model_name)
