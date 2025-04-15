import os
import random

import seaborn as sns
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, confusion_matrix, f1_score, roc_auc_score, RocCurveDisplay, roc_curve, auc
from collections import defaultdict
# Modèles de classification :
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

def plot_prediction_line_by_id(agg_df, threshold=0.5):
    # Ajoute la couleur selon la vraie classe
    agg_df["color"] = agg_df["true_label"].map({0: "blue", 1: "red"})
    agg_df["y"] = agg_df["true_label"]  # 0 en bas, 1 en haut pour séparer les points

    # Plot
    plt.figure(figsize=(12, 2.5))
    plt.scatter(agg_df["proba"], agg_df["y"], c=agg_df["color"], alpha=0.7)

    # Décorations
    plt.axvline(threshold, color='gray', linestyle='--', label=f"Seuil {threshold}")
    plt.yticks([0, 1], ["Classe réelle : 0", "Classe réelle : 1"])
    plt.xlabel("Probabilité prédite (par id_cas)")
    plt.title("Répartition des prédictions agrégées par id_cas")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend()


# Filtrer les données originales en fonction des id_cas attribués
def filter_by_ids(data, id_list):
    return [entry for entry in data if entry["id_cas"] in id_list]

def extract_XY(dataset):
    X = np.array([entry["embedding"] for entry in dataset])
    Y = np.array([entry["target"] for entry in dataset])
    return X, Y


def classifier_training(json_rep, json_file, model_name, seed, agg = 'median', threshold=0.4):
    json_path = os.path.join(json_rep, json_file+".json")
    with open(json_path) as f:
        data = json.load(f)

    output_dir = {"SVM": "SVM", "Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGBoost"}.get(model_name)
    os.makedirs(output_dir, exist_ok=True)


    # Groupement par id_cas
    id_cas_to_entries = defaultdict(list)
    for entry in data:
        id_cas_to_entries[entry["id_cas"]].append(entry)

    # Associer une target à chaque id_cas
    id_cas_to_target = {id_cas: entries[0]["target"] for id_cas, entries in id_cas_to_entries.items()}

    # Étape d’équilibrage des id_cas avant split
    ids_0 = [id_cas for id_cas, t in id_cas_to_target.items() if t == 0]
    ids_1 = [id_cas for id_cas, t in id_cas_to_target.items() if t == 1]
    n_balanced = min(len(ids_0), len(ids_1))

    np.random.seed(seed)
    ids_0_bal = np.random.choice(ids_0, n_balanced, replace=False).tolist()
    ids_1_bal = np.random.choice(ids_1, n_balanced, replace=False).tolist()
    balanced_ids = ids_0_bal + ids_1_bal
    np.random.shuffle(balanced_ids)

    # Mise à jour du mapping équilibré
    id_targets = [id_cas_to_target[i] for i in balanced_ids]

    # Split en train / temp, puis temp -> val + test
    ids_train, ids_test = train_test_split(
        balanced_ids, test_size=0.15, stratify=id_targets, random_state=seed
    )

    train_data = filter_by_ids(data, ids_train)
    test_data = filter_by_ids(data, ids_test)
    print("ID des cas dans le set de test :", ids_test)

    X_train, Y_train = extract_XY(train_data)
    X_test, Y_test = extract_XY(test_data)


    # Entraînement du modèle
    if model_name == "SVM":
        model = SVC(kernel='linear',
                    probability=True,
                    random_state=seed)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=seed,
                                   max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100,
                                       max_depth=100,
                                       random_state=seed)
    elif model_name == "XGBoost":
        model = XGBClassifier(dart_normalized_type = "forest",
                              learning_rate = 0.05,
                              max_iterations = 50,
                              max_depth = 10,
                              use_label_encoder=False,
                              eval_metric='logloss',
                              random_state=seed)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    Y_proba_all = model.predict_proba(X_test)
    Y_proba = Y_proba_all[:,1]


    # ---------- Evaluation --------------

    recall_before = recall_score(Y_test, Y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    precision_before = tp / (tp + fp)
    specificity_before = tn / (tn + fp)
    f1_before = f1_score(Y_test, Y_pred)
    cm_before = confusion_matrix(Y_test, Y_pred)


    # *********************** TEST AVEC AGGREGATION PAR ID ******************************

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
        "proba": agg,
        "true_label": "first"  # on suppose que le label est le même pour tous les vecteurs du cas
    }).reset_index()

    # ---- Seuil à 0.5 pour faire la prédiction binaire par id_cas ----
    agg_df["pred_label"] = (agg_df["proba"] >= threshold).astype(int)

    # ---- Évaluation par ID ----
    y_true = agg_df["true_label"]
    y_pred = agg_df["pred_label"]
    y_score = agg_df["proba"]


    # ---- Métriques ----

    recall_after = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity_after = tn / (tn + fp)
    f1_after = f1_score(y_true, y_pred)
    precision_after = tp / (tp + fp)

    # ----- Courbe ROC + affichage des 2 courbes --------

    fpr_test, tpr_test, _ = roc_curve(Y_test, Y_proba)
    roc_auc_test = auc(fpr_test, tpr_test)

    fpr_agg, tpr_agg, _ = roc_curve(y_true, y_score)
    roc_auc_agg = auc(fpr_agg, tpr_agg)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr_test, tpr_test, color='orange', lw=2, label=f"Avant agrégation (AUC = {roc_auc_test:.2f})")
    plt.plot(fpr_agg, tpr_agg, color='purple', lw=2, label=f"Après agrégation (AUC = {roc_auc_agg:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classifieur aléatoire')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs', fontsize=12)
    plt.ylabel('Sensibilité', fontsize=12)
    plt.title('Courbe ROC - Comparaison avant/après agrégation', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # -------------- Pour l'enregistrement des différents fichiers --------------
    list_metrics = f"R{int(recall_after*100)}_S{int(specificity_after*100)}_AUC{int(roc_auc_agg*100)}"
    output_dir_sub = f"{output_dir}/{json_file}/{list_metrics}_seed{seed}"
    os.makedirs(output_dir_sub, exist_ok=True)
    # ---------------------------------------------------------------------------

    output_path_plot = os.path.join(output_dir_sub, f"{json_file}_courbe_ROC_{list_metrics}.png")
    plt.savefig(output_path_plot, dpi=300)
    plt.show()

    plot_prediction_line_by_id(agg_df, threshold=threshold)
    output_path_plot = os.path.join(output_dir_sub, f"{json_file}_droite_repartition_{list_metrics}.png")
    plt.savefig(output_path_plot, dpi=300)
    plt.show()


    # ---- Matrice de confusion agrégée + affichage des 2 matrices ----

    cm_after = confusion_matrix(y_true, y_pred)
    labels = ['Non STEMI', 'STEMI']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm_before, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[0], annot_kws={"size": 14}, vmin=0)
    axes[0].set_title("Avant agrégation", fontsize=14)
    axes[0].set_xlabel("Prédiction", fontsize=12)
    axes[0].set_ylabel("Vérité terrain", fontsize=12)

    text_before = (f"Sensibilité (TP / (TP + FN)) : {recall_before:.2f}"
                   f"\nSpécificité (TN / (TN + FP)) : {specificity_before:.2f}"
                    f"\nPrécision (TP / (TP + FP)) : {precision_before:.2f}"
                   f"\nF1 Score : {f1_before:.2f}")
    axes[0].text(0.5, -0.25, text_before, fontsize=12, ha='center', va='top', transform=axes[0].transAxes)

    sns.heatmap(cm_after, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels, yticklabels=labels, ax=axes[1], annot_kws={"size": 14}, vmin=0)
    axes[1].set_title("Après agrégation par id_cas", fontsize=14)
    axes[1].set_xlabel("Prédiction", fontsize=12)
    axes[1].set_ylabel("Vérité terrain", fontsize=12)

    text_after = (f"Sensibilité (TP / (TP + FN)) : {recall_after:.2f}"
                  f"\nSpécificité (TN / (TN + FP)) : {specificity_after:.2f}"
                 f"\nPrécision (TP / (TP + FP)) : {precision_after:.2f}"

                  f"\nF1 Score : {f1_after:.2f}")
    axes[1].text(0.5, -0.25, text_after, fontsize=12, ha='center', va='top', transform=axes[1].transAxes)

    plt.tight_layout()

    output_path_matrix = os.path.join(output_dir_sub, f"{json_file}_confusion_matrix_{list_metrics}.png")

    plt.savefig(output_path_matrix, dpi=300)  # dpi=300 pour une bonne qualité
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

    # Agrégation par moyenne ou mediane
    agg_df = df_test.groupby("id_cas").agg({
        "proba_0": agg,  # ou "median"
        "proba_1": agg,  # ou "median"
        "true_label": "first"
    }).reset_index()

    # Prédiction finale (selon proba_1 >= 0.5)
    agg_df["pred_label"] = (agg_df["proba_1"] >= threshold).astype(int)

    output_filename_csv = f"{json_file}_probabilities_{list_metrics}.csv"
    output_path_csv = os.path.join(output_dir_sub, output_filename_csv)

    agg_df.to_csv(output_path_csv, index=False)

    if model_name == "SVM":
        joblib.dump(model, f"{output_dir_sub}/{json_file}_SVM_model_{list_metrics}.joblib")
    elif model_name == "Logistic Regression":
        joblib.dump(model, f"{output_dir_sub}/{json_file}_LR_model_{list_metrics}.joblib")
    elif model_name == "Random Forest":
        joblib.dump(model, f"{output_dir_sub}/{json_file}_RF_model_{list_metrics}.joblib")


    print(f"Résultats sauvegardés dans {output_dir_sub}")



if __name__ == "__main__":
    json_rep = "json_bert/camembertav2-base_raw_speaker_json"
    data_file = 'camembertav2-base_sans_metadata_3'
    # LR : "Logistic Regression"
    # RF : "Random Forest"
    # SVM : "SVM"
    # XGBoost
    seed = random.randint(1, 10000)
    model = "XGBoost"
    classifier_training(json_rep, data_file, model, seed)
