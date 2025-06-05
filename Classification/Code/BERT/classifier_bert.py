import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# import torch
from scipy.special import expit
from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_curve, auc, accuracy_score, precision_score
import joblib
import csv
import random
from tqdm import tqdm


# Plot la répartition des données en sortie pour savoir si elles sont "décisives" ou non
def plot_prediction_line_by_id(agg_df, threshold=0.5, output_path_plot_line=None):
    if agg_df.empty:
        print("agg_df is empty in plot_prediction_line_by_id, skipping plot.")
        return

    agg_df_copy = agg_df.copy()
    agg_df_copy["color"] = agg_df_copy["true_label"].map({0: "blue", 1: "red"})
    agg_df_copy["y"] = agg_df_copy["true_label"]

    plt.figure(figsize=(12, 2.5))
    plt.scatter(agg_df_copy["proba"], agg_df_copy["y"], c=agg_df_copy["color"], alpha=0.7)
    plt.axvline(threshold, color='gray', linestyle='--', label=f"Seuil {threshold}")
    plt.yticks([0, 1], ["Classe réelle : 0", "Classe réelle : 1"])
    plt.xlabel("Probabilité prédite (par id_cas)")
    plt.title("Répartition des prédictions agrégées par id_cas")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend()
    if output_path_plot_line:
        plt.savefig(output_path_plot_line, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()


# Filtrer les données originales en fonction des id_cas attribués
def filter_by_ids(data, id_list):
    return [entry for entry in data if entry["id_cas"] in id_list]


def extract_XY(dataset):
    X = np.array([entry["embedding"] for entry in dataset])
    Y = np.array([entry["target"] for entry in dataset])
    return X, Y


def evaluate_segment(df_segment, segment_name, segment_value, output_dir_sub, json_file_name_prefix,
                     global_metrics_suffix, base_labels):

    # Évalue les performances sur un segment de données spécifique et sauvegarde la matrice de confusion.

    if df_segment.empty or len(df_segment) < 1:  # Ajout vérification < 1
        print(
            f"  Skipping evaluation for segment {segment_name}={segment_value}: DataFrame is empty or too small ({len(df_segment)}).")
        return {}

    y_true = df_segment["true_label"]
    y_pred = df_segment["pred_label"]

    if len(y_true.unique()) < 1:
        print(f"  Skipping evaluation for segment {segment_name}={segment_value}: No true labels.")
        return {}

    # Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)

    cm_segment = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Gérer le cas où cm_segment.ravel() ne retourne pas 4 valeurs (par exemple si une classe est totalement absente dans y_true ET y_pred pour ce segment)
    if cm_segment.shape == (2, 2) and cm_segment.size == 4:
        tn, fp, fn, tp = cm_segment.ravel()
    elif len(y_true.unique()) == 1:
        # Si une seule classe dans y_true, on initialise pour éviter des erreurs de division par zéro plus tard
        # et on remplit en fonction des prédictions.
        if y_true.unique()[0] == 0:  # Uniquement classe 0 (Négatifs)
            tn = (y_pred == 0).sum()
            fp = (y_pred == 1).sum()
            fn = 0
            tp = 0
        else:  # Uniquement classe 1 (Positifs)
            tn = 0
            fp = 0
            fn = (y_pred == 0).sum()
            tp = (y_pred == 1).sum()
    else:  # Cas où la matrice n'est pas 2x2, mais y_true est mixte (rare si y_true et y_pred sont binaires)
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)  # VPP
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc_segment = 0.5  # Défaut
    if "proba" in df_segment.columns and len(y_true.unique()) >= 2:
        y_proba = df_segment["proba"]  # Probabilité de la classe positive
        fpr_seg, tpr_seg, _ = roc_curve(y_true, y_proba)
        roc_auc_segment = auc(fpr_seg, tpr_seg)
    elif "proba" not in df_segment.columns:
        print(f"  Warning for segment {segment_name}={segment_value}: 'proba' column missing for ROC AUC.")
    elif len(y_true.unique()) < 2 and len(y_true) > 0:  # Si qu'une classe mais des données
        print(
            f"  Warning for segment {segment_name}={segment_value}: Only one class in y_true ({y_true.unique()}), ROC AUC is not well-defined (set to 0.5).")

    segment_metrics = {
        "accuracy": accuracy, "recall": recall, "specificity": specificity,
        "precision": precision, "f1": f1, "auc": roc_auc_segment,
        "n_cases": len(df_segment)
    }


    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm_segment, annot=True, fmt='d', cmap='Blues',
                xticklabels=base_labels, yticklabels=base_labels,
                annot_kws={"size": 14}, vmin=0)

    title = (f"Matrice de Confusion - {segment_name}: {str(segment_value)}\n"
             f"({segment_metrics['n_cases']} cas, AUC={segment_metrics['auc']:.2f})")
    plt.title(title, fontsize=13)
    plt.xlabel("Prédiction", fontsize=12)
    plt.ylabel("Vérité terrain", fontsize=12)

    text_metrics_plot = (
        f"Accuracy: {segment_metrics['accuracy']:.2f}\nSensibilité: {segment_metrics['recall']:.2f}\nSpécificité: {segment_metrics['specificity']:.2f}"
        f"\nPrécision: {segment_metrics['precision']:.2f}\nF1 Score: {segment_metrics['f1']:.2f}")
    plt.text(0.5, -0.30, text_metrics_plot, fontsize=11, ha='center', va='top', transform=plt.gca().transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    safe_segment_value = str(segment_value).replace('<', 'lt_').replace('>', 'gt_').replace(' ', '_').replace('-',
                                                                                                              '_')
    filename = f"{json_file_name_prefix}_CM_segment_{segment_name}_{safe_segment_value}_{global_metrics_suffix}.png"
    output_path = os.path.join(output_dir_sub, filename)
    try:
        plt.savefig(output_path, dpi=300)
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
    plt.close()
    return segment_metrics


def classifier_training(json_rep, json_file, model_name, seed, agg='median', threshold=0.5):
    json_path = os.path.join(json_rep, json_file + ".json")
    try:
        with open(json_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found at {json_path}")
        return 0, 0, 0, 0, 0, {}  # Ajout d'un retour pour les métriques segmentées

    if not data:
        print(f"ERROR: JSON file {json_path} is empty.")
        return 0, 0, 0, 0, 0, {}

    # Vérification de la présence des clés 'sexe' et 'age'
    has_sex_age_info = "sexe" in data[0] and "age" in data[0]
    if not has_sex_age_info:
        print(
            f"Warning for {json_file}: 'sexe' or 'age' key not found in the first data entry. Segmented analysis will be skipped.")

    output_dir_name = {"SVM": "../../Results/BERT/SVM", "Naive Bayes": "../../Results/BERT/NB", "Logistic Regression": "../../Results/BERT/LR",
                       "Random Forest": "../../Results/BERT/RF", "XGBoost": "../../Results/BERT/XGBoost", "Ridge": "../../Results/BERT/RidgeClassifier",
                       "Catboost": "../../Results/BERT/Catboost", "AdaBoost": "../../Results/BERT/AdaBoost"}.get(model_name)
    if output_dir_name is None:
        print(f"ERROR: Model name {model_name} not recognized for output directory.")
        return 0, 0, 0, 0, 0, {}

    id_cas_to_entries = defaultdict(list)
    for entry in data:
        id_cas_to_entries[entry["id_cas"]].append(entry)

    id_cas_to_target = {id_cas: entries[0]["target"] for id_cas, entries in id_cas_to_entries.items()}
    # Extraire sexe et âge pour chaque id_cas
    id_cas_to_sex = {}
    id_cas_to_age = {}
    if has_sex_age_info:
        id_cas_to_sex = {id_cas: entries[0].get("sexe", "N/A") for id_cas, entries in id_cas_to_entries.items()}
        id_cas_to_age = {id_cas: entries[0].get("age", np.nan) for id_cas, entries in
                         id_cas_to_entries.items()}  # Utiliser np.nan si age est manquant (normalement non)

    ids_0 = [id_cas for id_cas, t in id_cas_to_target.items() if t == 0]
    ids_1 = [id_cas for id_cas, t in id_cas_to_target.items() if t == 1]

    if not ids_0 or not ids_1:
        print(
            f"Warning for seed {seed}: Not enough samples for one or both classes to balance. ids_0: {len(ids_0)}, ids_1: {len(ids_1)}. Skipping.")
        return 0, 0, 0, 0, 0, {}
    n_balanced = min(len(ids_0), len(ids_1))
    if n_balanced == 0:
        print(f"Warning for seed {seed}: n_balanced is 0. Skipping.")
        return 0, 0, 0, 0, 0, {}
    np.random.seed(seed)
    ids_0_bal = np.random.choice(ids_0, n_balanced, replace=False).tolist()
    ids_1_bal = np.random.choice(ids_1, n_balanced, replace=False).tolist()
    balanced_ids = ids_0_bal + ids_1_bal
    np.random.shuffle(balanced_ids)
    id_targets = [id_cas_to_target[i] for i in balanced_ids]

    min_samples_per_class_for_stratify = 2
    if n_balanced < min_samples_per_class_for_stratify:
        print(
            f"Warning for seed {seed}: n_balanced per class ({n_balanced}) is less than {min_samples_per_class_for_stratify}. Stratified split might fail or be uninformative. Skipping.")
        return 0, 0, 0, 0, 0, {}

    try:
        ids_train, ids_test = train_test_split(balanced_ids, test_size=0.15, stratify=id_targets, random_state=seed)
    except ValueError as e:
        print(
            f"Error during train_test_split for seed {seed}: {e}. n_balanced: {n_balanced}, id_targets counts: {np.unique(id_targets, return_counts=True)}. Skipping.")
        return 0, 0, 0, 0, 0, {}
    if not ids_train or not ids_test:
        print(f"Warning for seed {seed}: ids_train or ids_test is empty after split. Skipping.")
        return 0, 0, 0, 0, 0, {}

    train_data = filter_by_ids(data, ids_train)
    test_data = filter_by_ids(data, ids_test)

    if not train_data or not test_data:
        print(f"Warning for seed {seed}: train_data or test_data is empty. Skipping.")
        return 0, 0, 0, 0, 0, {}

    X_train, Y_train = extract_XY(train_data)
    X_test, Y_test = extract_XY(test_data)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Warning for seed {seed}: X_train or X_test is empty. Skipping.")
        return 0, 0, 0, 0, 0, {}
    if len(np.unique(Y_train)) < 2:
        print(f"Warning for seed {seed}: Y_train has only one class: {np.unique(Y_train)}. Skipping.")
        return 0, 0, 0, 0, 0, {}
    if len(np.unique(Y_test)) < 2:
        print(
            f"Warning for seed {seed}: Y_test has only one class: {np.unique(Y_test)}. Metrics 'before' might be ill-defined. Skipping some evaluations.")

    # Entraînement du modèle
    if model_name == "SVM":
        model = SVC(kernel='poly', degree=4, probability=True, random_state=seed)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=seed, max_iter=10000, solver="liblinear")
    elif model_name == "Random Forest":
        model = RandomForestClassifier(max_depth=100, criterion="entropy", random_state=seed, max_features="sqrt")
    # elif model_name == "XGBoost":
    #     model = XGBClassifier(booster="gbtree", device="cuda" if torch.cuda.is_available() else "cpu",
    #                           learning_rate=0.05, max_depth=10, eval_metric='logloss', random_state=seed)
    # elif model_name == "Catboost":
    #     model = CatBoostClassifier(task_type="GPU" if torch.cuda.is_available() else "CPU", iterations=100, depth=6,
    #                                verbose=0, random_seed=seed)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Ridge":
        model = RidgeClassifier(random_state=seed)
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train, Y_train)
    Y_pred_before = model.predict(X_test)

    if hasattr(model, "decision_function") and not (model_name == "SVM" and model.probability):  # SVM avec proba=False
        scores = model.decision_function(X_test)
        if scores.ndim == 1:
            proba_pos = expit(scores); Y_proba_all = np.column_stack([1 - proba_pos, proba_pos])
    elif hasattr(model, "predict_proba"):
        Y_proba_all = model.predict_proba(X_test)
    else:  # Cas comme RidgeClassifier sans predict_proba natif
        Y_proba_all = np.zeros((len(Y_test), 2))
        Y_proba_all[:, 0] = 1 - Y_pred_before
        Y_proba_all[:, 1] = Y_pred_before

    Y_proba_before = Y_proba_all[:, 1]

    # ---------- Evaluation Avant Agrégation --------------
    recall_before = 0.0;
    precision_before = 0.0;
    specificity_before = 0.0;
    f1_before = 0.0
    cm_before_heatmap = np.zeros((2, 2), dtype=int)

    if len(np.unique(Y_test)) >= 1:
        cm_before_values = confusion_matrix(Y_test, Y_pred_before, labels=[0, 1]).ravel()
        if cm_before_values.size == 4:
            tn_b, fp_b, fn_b, tp_b = cm_before_values
            recall_before = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
            precision_before = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
            specificity_before = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
            f1_before = 2 * (precision_before * recall_before) / (precision_before + recall_before) if (precision_before + recall_before) > 0 else 0.0
            cm_before_heatmap = confusion_matrix(Y_test, Y_pred_before, labels=[0, 1])
        else:  # Devrait pas arriver avec labels=[0,1]
            print(f"Warning for seed {seed}: Confusion matrix 'before' not 2x2.")

    # *********************** TEST AVEC AGGREGATION PAR ID ******************************
    id_cas_test_embeddings = [entry["id_cas"] for entry in test_data]

    df_test_agg_data = {
        "id_cas": id_cas_test_embeddings,
        "true_label": Y_test,
        "proba": Y_proba_before
    }
    # Ajouter sexe et age si disponibles, en mappant depuis id_cas
    if has_sex_age_info:
        df_test_agg_data["sexe"] = [id_cas_to_sex.get(id_val, "N/A") for id_val in id_cas_test_embeddings]
        df_test_agg_data["age"] = [id_cas_to_age.get(id_val, np.nan) for id_val in id_cas_test_embeddings]

    df_test_agg = pd.DataFrame(df_test_agg_data)

    if df_test_agg.empty:
        print(f"Warning for seed {seed}: df_test_agg is empty before aggregation. Skipping.")
        return 0, 0, 0, 0, 0, {}

    # Agrégation
    agg_dict = {"proba": agg, "true_label": "first"}
    if has_sex_age_info:
        agg_dict["sexe"] = "first"
        agg_dict["age"] = "first"

    agg_df = df_test_agg.groupby("id_cas").agg(agg_dict).reset_index()

    if agg_df.empty:
        print(f"Warning for seed {seed}: agg_df is empty after grouping. Skipping.")
        return 0, 0, 0, 0, 0, {}

    agg_df["pred_label"] = (agg_df["proba"] >= threshold).astype(int)
    agg_df_for_roc = agg_df.copy()

    total_cases_aggregated = len(agg_df)
    ambiguous_mask = agg_df["proba"].between(0.45, 0.55)  # Seuil d'ambiguïté
    nb_removed_ambiguous = ambiguous_mask.sum()
    agg_df_filtered = agg_df[~ambiguous_mask].copy()  # .copy() pour éviter SettingWithCopyWarning
    nb_kept_for_metrics = len(agg_df_filtered)

    recall_after = 0.0;
    specificity_after = 0.0;
    precision_after = 0.0;
    f1_after = 0.0
    roc_auc_agg = 0.5
    fpr_agg, tpr_agg = np.array([0, 1]), np.array([0, 1])
    cm_after_heatmap = np.zeros((2, 2), dtype=int)

    if not agg_df_filtered.empty and len(agg_df_filtered["true_label"].unique()) >= 1:
        y_true_filtered = agg_df_filtered["true_label"]
        y_pred_filtered = agg_df_filtered["pred_label"]

        cm_after_heatmap = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[0, 1])
        if cm_after_heatmap.size == 4:
            tn_agg, fp_agg, fn_agg, tp_agg = cm_after_heatmap.ravel()
            recall_after = tp_agg / (tp_agg + fn_agg) if (tp_agg + fn_agg) > 0 else 0.0
            specificity_after = tn_agg / (tn_agg + fp_agg) if (tn_agg + fp_agg) > 0 else 0.0
            precision_after = tp_agg / (tp_agg + fp_agg) if (tp_agg + fp_agg) > 0 else 0.0
            f1_after = 2 * (precision_after * recall_after) / (precision_after + recall_after) if (precision_after + recall_after) > 0 else 0.0

        if len(y_true_filtered.unique()) < 2:
            print(
                f"Warning for seed {seed}: agg_df_filtered has only one class after filtering. Some 'after' metrics might be limited.")
    else:
        print(
            f"Warning for seed {seed}: After removing ambiguous cases, agg_df_filtered is empty or has no classes. 'After' metrics will be 0.")

    if not agg_df_for_roc.empty and len(agg_df_for_roc["true_label"].unique()) >= 2:
        fpr_agg, tpr_agg, _ = roc_curve(agg_df_for_roc["true_label"], agg_df_for_roc["proba"])
        roc_auc_agg = auc(fpr_agg, tpr_agg)
    else:
        if agg_df_for_roc.empty:
            print(f"Seed {seed}: agg_df_for_roc is empty. ROC AUC for 'after' uses default 0.5.")
        else:
            print(
                f"Seed {seed}: agg_df_for_roc has true_labels: {agg_df_for_roc['true_label'].unique()}. ROC AUC for 'after' uses default 0.5.")

    fpr_test, tpr_test = np.array([0, 1]), np.array([0, 1])  # default
    roc_auc_test = 0.5  # default
    if len(np.unique(Y_test)) >= 2:
        fpr_test, tpr_test, _ = roc_curve(Y_test, Y_proba_before)
        roc_auc_test = auc(fpr_test, tpr_test)
    else:
        print(f"Seed {seed}: Y_test has < 2 unique classes. ROC AUC for 'before' uses default 0.5.")

    # PLOTTING SECTION
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, color='orange', lw=2, label=f"Avant agrégation (AUC = {roc_auc_test:.2f})")
    plt.plot(fpr_agg, tpr_agg, color='purple', lw=2, label=f"Après agrégation (AUC = {roc_auc_agg:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classifieur aléatoire')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs');
    plt.ylabel('Sensibilité')
    plt.title('Courbe ROC - Comparaison avant/après agrégation');
    plt.legend(loc="lower right");
    plt.grid(True);
    plt.tight_layout()

    list_metrics_suffix = f"R{int(recall_after * 100)}_S{int(specificity_after * 100)}_P{int(precision_after * 100)}_F1{int(f1_after * 100)}_AUC{int(roc_auc_agg * 100)}"
    output_dir_sub = os.path.join(output_dir_name, json_file, f"{list_metrics_suffix}_seed{seed}")
    os.makedirs(output_dir_sub, exist_ok=True)

    output_path_plot_roc = os.path.join(output_dir_sub, f"{json_file}_courbe_ROC_{list_metrics_suffix}.png")
    plt.savefig(output_path_plot_roc, dpi=300);
    plt.close()

    output_path_plot_line = os.path.join(output_dir_sub, f"{json_file}_droite_repartition_{list_metrics_suffix}.png")
    plot_prediction_line_by_id(agg_df_for_roc, threshold=threshold,
                               output_path_plot_line=output_path_plot_line)  # Appel avec sauvegarde

    labels_cm = ['Non STEMI', 'STEMI']
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(14, 7))
    sns.heatmap(cm_before_heatmap, annot=True, fmt='d', cmap='Oranges', xticklabels=labels_cm, yticklabels=labels_cm,
                ax=axes_cm[0], annot_kws={"size": 14}, vmin=0)
    axes_cm[0].set_title("Avant agrégation");
    axes_cm[0].set_xlabel("Prédiction");
    axes_cm[0].set_ylabel("Vérité terrain")
    text_before_metrics = (
        f"Sensibilité: {recall_before:.2f}\nSpécificité: {specificity_before:.2f}\nPrécision: {precision_before:.2f}\nF1 Score: {f1_before:.2f}")
    axes_cm[0].text(0.5, -0.28, text_before_metrics, fontsize=11, ha='center', va='top',
                    transform=axes_cm[0].transAxes)

    sns.heatmap(cm_after_heatmap, annot=True, fmt='d', cmap='Purples', xticklabels=labels_cm, yticklabels=labels_cm,
                ax=axes_cm[1], annot_kws={"size": 14}, vmin=0)
    axes_cm[1].set_title("Après agrégation par id_cas");
    axes_cm[1].set_xlabel("Prédiction");
    axes_cm[1].set_ylabel("Vérité terrain")
    text_after_metrics = (
        f"Sur {total_cases_aggregated} cas agrégés:\n  - Cas ambigus (0.45-0.55) retirés: {nb_removed_ambiguous}\n  - Cas conservés pour métriques: {nb_kept_for_metrics}\n\nMétriques sur cas conservés:\nSensibilité: {recall_after:.2f}\nSpécificité: {specificity_after:.2f}\nPrécision: {precision_after:.2f}\nF1 Score: {f1_after:.2f}")
    axes_cm[1].text(0.5, -0.28, text_after_metrics, fontsize=11, ha='center', va='top',
                    transform=axes_cm[1].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path_matrix = os.path.join(output_dir_sub, f"{json_file}_confusion_matrix_{list_metrics_suffix}.png")
    plt.savefig(output_path_matrix, dpi=300);
    plt.close()

    # *********** ÉVALUATION PAR SEGMENT *************
    segmented_metrics_results = {}  # Pour stocker les métriques par segment
    if has_sex_age_info and not agg_df_filtered.empty:

        # Par Sexe
        if "sexe" in agg_df_filtered.columns:
            for sex_value in agg_df_filtered["sexe"].unique():
                if pd.isna(sex_value) or sex_value == "N/A": continue  # Ignorer les valeurs manquantes/N.A.
                df_segment_sex = agg_df_filtered[agg_df_filtered["sexe"] == sex_value]
                if not df_segment_sex.empty:
                    metrics = evaluate_segment(df_segment_sex, "sexe", sex_value, output_dir_sub, json_file,
                                               list_metrics_suffix, labels_cm)
                    segmented_metrics_results[f"sexe_{sex_value}"] = metrics

        # Par Tranche d'âge
        if "age" in agg_df_filtered.columns:
            # S'assurer que 'age' est numérique, convertir les erreurs en NaN
            agg_df_filtered['age'] = pd.to_numeric(agg_df_filtered['age'], errors='coerce')

            age_bins = [0, 39, 60, np.inf]  # <40, 40-60, >60
            age_labels = ["<40 ans", "40-60 ans", ">60 ans"]
            agg_df_filtered["tranche_age"] = pd.cut(agg_df_filtered["age"], bins=age_bins, labels=age_labels,
                                                    right=True)  # right=True inclut la borne sup de l'intervalle précédent

            for age_group_value in age_labels:
                df_segment_age = agg_df_filtered[agg_df_filtered["tranche_age"] == age_group_value]
                if not df_segment_age.empty:
                    metrics = evaluate_segment(df_segment_age, "tranche_age", age_group_value, output_dir_sub,
                                               json_file, list_metrics_suffix, labels_cm)
                    segmented_metrics_results[
                        f"age_{age_group_value.replace(' ', '_').replace('<', 'lt').replace('>', 'gt')}"] = metrics  # Nom de clé safe
    else:
        if not has_sex_age_info: print("Skipping segmented analysis: 'sexe' or 'age' info not available.")
        if agg_df_filtered.empty: print("Skipping segmented analysis: agg_df_filtered is empty.")

    # Sauvegarde CSV
    proba_0_all = Y_proba_all[:, 0]
    proba_1_all = Y_proba_all[:, 1]
    df_for_csv_individual = pd.DataFrame(
        {"id_cas": id_cas_test_embeddings, "true_label": Y_test, "proba_0_individual": proba_0_all,
         "proba_1_individual": proba_1_all})

    # Utiliser agg_df_for_roc car il contient tous les cas agrégés (avant filtrage ambigu) et les infos sexe/age si dispo
    df_for_csv_aggregated = agg_df_for_roc.copy()  # Contient déjà id_cas, proba, true_label, pred_label, et potentiellement sexe, age
    df_for_csv_aggregated.rename(
        columns={"proba": "proba_1_agg", "true_label": "true_label_cas", "pred_label": "pred_label_cas"}, inplace=True)
    df_for_csv_aggregated["proba_0_agg"] = 1 - df_for_csv_aggregated["proba_1_agg"]

    # Réorganiser les colonnes pour la lisibilité
    cols_order = ["id_cas", "true_label_cas", "proba_0_agg", "proba_1_agg", "pred_label_cas"]
    if has_sex_age_info and "sexe" in df_for_csv_aggregated.columns: cols_order.append("sexe")
    if has_sex_age_info and "age" in df_for_csv_aggregated.columns: cols_order.append("age")

    # S'assurer que toutes les colonnes existent avant de réindexer
    cols_order = [col for col in cols_order if col in df_for_csv_aggregated.columns]
    df_for_csv_aggregated = df_for_csv_aggregated[cols_order]

    output_filename_csv = f"{json_file}_probabilities_agg_{list_metrics_suffix}.csv"
    output_path_csv = os.path.join(output_dir_sub, output_filename_csv)
    df_for_csv_aggregated.to_csv(output_path_csv, index=False)

    # Sauvegarde du modèle
    model_filename = f"{json_file}_{model_name.replace(' ', '')}_model_{list_metrics_suffix}.joblib"
    joblib.dump(model, os.path.join(output_dir_sub, model_filename))

    return (int(recall_after * 100), int(specificity_after * 100),
            int(precision_after * 100), int(f1_after * 100),
            int(roc_auc_agg * 100), segmented_metrics_results)


if __name__ == "__main__":
    models = ["Naive Bayes", "Random Forest", "Ridge"]
    json_reps = {
        r"../../../Vectorization/Results/BERT/nn_camembertav2-base_nlp_json": "camembertav2-base_sans_metadata_3"}
    main_csv_file = "../../Results/BERT/result_bert.csv"

    # Définition les métriques pour chaque segment
    segment_metric_keys = ["accuracy", "recall", "specificity", "precision", "f1", "auc", "n_cases"]

    # Construction des noms des colonnes pour le CSV final
    header_row = ["model", "json_rep_path", "json_file_key", "seed",
                  "sensibility", "specificity", "precision", "f1_score", "auc_roc"]

    sex_labels_for_csv = ["M", "F"]
    for sex_val in sex_labels_for_csv:
        for metric_key in segment_metric_keys:
            header_row.append(f"sex_{sex_val}_{metric_key}")

    age_labels_for_csv = ["lt40_ans", "40-60_ans", "gt60_ans"]
    for age_key_suffix in age_labels_for_csv:
        for metric_key in segment_metric_keys:
            header_row.append(f"age_{age_key_suffix}_{metric_key}")

    with open(main_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header_row)

    with open(main_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for model_name_loop in models:
            for json_rep_path, json_file_key in json_reps.items():
                print(f"Treating '{json_file_key}' from '{json_rep_path}' with model '{model_name_loop}'")
                success_count = 0
                for i in tqdm(range(70), desc=f"{model_name_loop} - {json_file_key}"): # 70 tests
                    seed = random.randint(1, 100000)
                    sens, spe, pre, f1, roc, segmented_metrics = classifier_training(
                        json_rep_path, json_file_key, model_name_loop, seed, agg='median'
                    )

                    if not (sens == 0 and spe == 0 and pre == 0 and f1 == 0 and roc == 50):
                        row_data = [model_name_loop, json_rep_path, json_file_key, seed, sens, spe, pre, f1, roc]

                        for sex_val in sex_labels_for_csv:
                            segment_data = segmented_metrics.get(f"sexe_{sex_val}", {})
                            for metric_key in segment_metric_keys:
                                value = segment_data.get(metric_key, np.nan)
                                if metric_key == "n_cases":
                                    row_data.append(int(value) if pd.notna(value) else np.nan)
                                else:
                                    row_data.append(round(value * 100) if pd.notna(value) else np.nan)

                        for age_key_suffix in age_labels_for_csv:
                            segment_data = segmented_metrics.get(f"age_{age_key_suffix}", {})
                            for metric_key in segment_metric_keys:
                                value = segment_data.get(metric_key, np.nan)
                                if metric_key == "n_cases":
                                    row_data.append(int(value) if pd.notna(value) else np.nan)
                                else:
                                    row_data.append(round(value * 100) if pd.notna(value) else np.nan)

                        writer.writerow(row_data)
                        success_count += 1
                    else:
                        print(
                            f"Seed {seed} for {model_name_loop} on {json_file_key} resulted in default/error metrics, not written to CSV.")
                print(
                    f"Entraînement réalisé pour {model_name_loop} sur {json_file_key}! {success_count} itérations réussies.")

    print("Terminé!")
