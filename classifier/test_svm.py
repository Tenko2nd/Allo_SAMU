import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, recall_score, confusion_matrix, f1_score, roc_auc_score, RocCurveDisplay
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict


# Filtrer les données originales en fonction des id_cas attribués
def filter_by_ids(data, id_list):
    return [entry for entry in data if entry["id_cas"] in id_list]

def extract_XY(dataset):
    X = np.array([entry["embedding"] for entry in dataset])
    Y = np.array([entry["target"] for entry in dataset])
    return X, Y


def classifier_training(json_file):
    with open(json_file) as f:
        data = json.load(f)

    # Regrouper les entrées par id_cas
    id_cas_to_entries = defaultdict(list)
    for entry in data:
        id_cas_to_entries[entry["id_cas"]].append(entry)

    # Lister les id_cas uniques
    unique_ids = list(id_cas_to_entries.keys())

    # Associer une target à chaque id_cas (en supposant que toutes les entrées d’un même id_cas ont la même target)
    id_cas_to_target = {id_cas: entries[0]["target"] for id_cas, entries in id_cas_to_entries.items()}

    # Split sur les id_cas uniques (en stratifiant selon leur target)
    id_targets = [id_cas_to_target[i] for i in unique_ids]
    ids_train, ids_temp = train_test_split(unique_ids, test_size=0.30, stratify=id_targets, random_state=42)
    id_targets_temp = [id_cas_to_target[i] for i in ids_temp]
    ids_val, ids_test = train_test_split(ids_temp, test_size=0.5, stratify=id_targets_temp, random_state=42)


    train_data = filter_by_ids(data, ids_train)
    val_data = filter_by_ids(data, ids_val)
    test_data = filter_by_ids(data, ids_test)

    X_train, Y_train = extract_XY(train_data)
    X_val, Y_val = extract_XY(val_data)
    X_test, Y_test = extract_XY(test_data)


    svm_model = SVC(kernel = 'linear', probability=True, random_state=42)
    svm_model.fit(X_train, Y_train)

    Y_pred = svm_model.predict(X_test)
    Y_proba_all = svm_model.predict_proba(X_test)
    Y_proba = Y_proba_all[:,1]


    # ---------- Evaluation ----------------------
    print("********** Evaluation avant agrégation ************")

    score = recall_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)

    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn + fp)

    f1 = f1_score(Y_test, Y_pred)

    roc_auc = roc_auc_score(Y_test, Y_proba)


    print(f"Sensibilité : {score:.4f}")
    print(f"\nSpecificite : {specificity:.4f}")
    print(f"ROC AUC : {roc_auc:.4f}")


    print("\nClassification Report:")
    print(report)

    # ----- Matrice de confusion ---------
    cm = confusion_matrix(Y_test, Y_pred)
    cmap = plt.cm.Blues
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

    print(f"\nDétail TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    RocCurveDisplay.from_predictions(Y_test, Y_proba)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classifieur aléatoire')
    plt.title("ROC Curve (SVM)")
    plt.grid(True)
    plt.show()


    # ----------------- TEST AVEC AGGREGATION PAR ID ------------------------

    # ---- Etape : récupérer les id_cas associés à chaque vecteur de test ----
    id_cas_test = [entry["id_cas"] for entry in test_data]
    true_labels = [entry["target"] for entry in test_data]

    # ---- Construction d’un DataFrame pour faciliter l’agrégation ----
    df_test = pd.DataFrame({
        "id_cas": id_cas_test,
        "true_label": true_labels,
        "proba": Y_proba,
        "pred": Y_pred
    })

    # ---- Agrégation par ID (moyenne 'mean', médiane 'median' -> BEST : median) ----
    agg_df = df_test.groupby("id_cas").agg({
        "proba": "median",
        "true_label": "first"  # on suppose que le label est le même pour tous les vecteurs du cas
    }).reset_index()

    # ---- Seuil à 0.5 pour faire la prédiction binaire par id_cas ----
    agg_df["pred_label"] = (agg_df["proba"] >= 0.5).astype(int)

    # ---- Évaluation par ID ----
    y_true = agg_df["true_label"]
    y_pred = agg_df["pred_label"]
    y_score = agg_df["proba"]

    # ---- Métriques ----
    print("********** Evaluation APRES agrégation par ID ************")

    n_vecteurs_test = len(test_data)
    n_id_cas_test = len(set(id_cas_test))
    n_id_cas_agg = len(agg_df)

    print("\n------ Taille des données ------")
    print(f"Nombre de vecteurs dans le test set       : {n_vecteurs_test}")
    print(f"Nombre d'id_cas après agrégation (agg_df) : {n_id_cas_agg}")

    print(f"Sensibilité : {recall_score(y_true, y_pred):.4f}")
    print(
        f"Spécificité : {confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1]):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"AUC ROC : {roc_auc_score(y_true, y_score):.4f}")
    print("\nClassification Report :")
    print(classification_report(y_true, y_pred))


    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title("ROC Curve (SVM)")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classifieur aléatoire')
    plt.grid(True)
    plt.show()

    # ---- Matrice de confusion agrégée ----
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Non STEMI', 'STEMI']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.yticks(rotation=0, va="center")
    plt.xticks(rotation=45, ha="right")
    plt.title("Matrice de Confusion (par ID)")
    plt.tight_layout()
    plt.show()

    # ********* Sauvegarde des proba de test dans un .csv ***************

    # On reprend toutes les probabilités pour les deux classes
    proba_0 = Y_proba_all[:, 0]
    proba_1 = Y_proba_all[:, 1]

    # Nouveau DataFrame complet avec proba_0 et proba_1
    df_test = pd.DataFrame({
        "id_cas": id_cas_test,
        "true_label": true_labels,
        "proba_0": proba_0,
        "proba_1": proba_1
    })

    # Agrégation par moyenne (ou médiane si tu préfères)
    agg_df = df_test.groupby("id_cas").agg({
        "proba_0": "mean",  # ou "median"
        "proba_1": "mean",  # ou "median"
        "true_label": "first"
    }).reset_index()

    # Prédiction finale (selon proba_1 >= 0.5)
    agg_df["pred_label"] = (agg_df["proba_1"] >= 0.5).astype(int)

    # ---- Sauvegarde CSV ----
    output_dir = "SVM_probabilities"
    output_filename_csv = "SVM_probabilities_bert.csv"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename_csv)

    agg_df.to_csv(output_path, index=False)
    print(f"Résultats sauvegardés dans {output_path}")



if __name__ == "__main__":
    data_file = 'json_camembert/camembert_sans_metadata.json'
    pipeline = "camemBERT"
    classifier_training(data_file)
