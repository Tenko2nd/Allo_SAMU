import os
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


# Filtrer les données originales en fonction des id_cas attribués
def filter_by_ids(data, id_list):
    return [entry for entry in data if entry["id_cas"] in id_list]

def extract_XY(dataset):
    X = np.array([entry["embedding"] for entry in dataset])
    Y = np.array([entry["target"] for entry in dataset])
    return X, Y


def classifier_training(json_file, model_name, seed = 42):
    with open(json_file) as f:
        data = json.load(f)

    output_dir = None
    # ---- Sauvegarde CSV et plots futurs ----
    if model_name == "SVM":
        output_dir = "SVM"
    elif model_name == "Logistic Regression":
        output_dir = "LR"
    elif model_name == "Random Forest":
        output_dir = "RF"
    os.makedirs(output_dir, exist_ok=True)

    # ***************** GROUPEMENT DES ID ENSEMBLE ***********************

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
    test_data = filter_by_ids(data, ids_test)

    X_train, Y_train = extract_XY(train_data)
    X_test, Y_test = extract_XY(test_data)

    # ***************** ENTRAINEMENT DU MODELE ***********************


    if model_name == "SVM":
        model = SVC(kernel = 'linear', probability=True, random_state=seed)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=seed, class_weight='balanced', max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, class_weight='balanced')
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    Y_proba_all = model.predict_proba(X_test)
    Y_proba = Y_proba_all[:,1]


    # ---------- Evaluation --------------
    print("********** Evaluation avant agrégation ************")

    score = recall_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_proba)
    cm_before = confusion_matrix(Y_test, Y_pred)


    print(f"Sensibilité : {score:.4f}")
    print(f"Specificite : {specificity:.4f}")
    print(f"ROC AUC : {roc_auc:.4f}")
    print(f"F1 score : {f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print(f"\nDétail TN={tn}, FP={fp}, FN={fn}, TP={tp}")


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

    print(f"Sensibilité : {recall_score(y_true, y_pred):.4f}")
    print(
        f"Spécificité : {confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1]):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"AUC ROC : {roc_auc_score(y_true, y_score):.4f}")
    print("\nClassification Report :")
    print(classification_report(y_true, y_pred))

    # ----- Courbe ROC + affichage des 2 courbes --------
    output_path_plot = os.path.join(output_dir, "courbe_ROC.png")

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
    plt.savefig(output_path_plot, dpi=300)  # dpi=300 pour une bonne qualité
    plt.show()

    # ---- Matrice de confusion agrégée + affichage des 2 matrices ----
    output_path_plot = os.path.join(output_dir, "confusion_matrix.png")

    cm_after = confusion_matrix(y_true, y_pred)
    labels = ['Non STEMI', 'STEMI']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm_before, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[0], annot_kws={"size": 14})
    axes[0].set_title("Avant agrégation", fontsize=14)
    axes[0].set_xlabel("Prédiction", fontsize=12)
    axes[0].set_ylabel("Vérité terrain", fontsize=12)

    sns.heatmap(cm_after, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels, yticklabels=labels, ax=axes[1], annot_kws={"size": 14})
    axes[1].set_title("Après agrégation par id_cas", fontsize=14)
    axes[1].set_xlabel("Prédiction", fontsize=12)
    axes[1].set_ylabel("Vérité terrain", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path_plot, dpi=300)  # dpi=300 pour une bonne qualité
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
        "proba_0": "median",  # ou "median"
        "proba_1": "median",  # ou "median"
        "true_label": "first"
    }).reset_index()

    # Prédiction finale (selon proba_1 >= 0.5)
    agg_df["pred_label"] = (agg_df["proba_1"] >= 0.5).astype(int)

    output_filename_csv = "probabilities_bert.csv"
    output_path_csv = os.path.join(output_dir, output_filename_csv)

    agg_df.to_csv(output_path_csv, index=False)
    print(f"Résultats sauvegardés dans {output_path_csv}")



if __name__ == "__main__":
    data_file = 'json_camembert/camembert_A_05.json'
    # LR : "Logistic Regression"
    # RF : "Random Forest"
    # SVM : "SVM"
    model = "SVM"
    classifier_training(data_file, model)
