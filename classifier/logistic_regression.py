import os
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression # Modèle utilisé
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, confusion_matrix, f1_score, roc_auc_score, RocCurveDisplay
import numpy as np
import json
import csv # Nécessaire pour sauvegarder les probabilités
import matplotlib.pyplot as plt

def classifier_training_lr(json_file):
    with open(json_file) as f:
        data = json.load(f)

    X = np.array([entry["embedding"][0] for entry in data])
    Y = np.array([entry["target"] for entry in data])

    ids = np.array([entry["id_cas"] for entry in data])

    X_train, X_test, Y_train, Y_test, ids_train, ids_test = train_test_split(X, Y, ids, test_size=0.15, random_state=42, stratify=Y)


    log_reg_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)

    log_reg_model.fit(X_train, Y_train)
    print("Entraînement terminé.")


    Y_pred = log_reg_model.predict(X_test)
    Y_proba_all = log_reg_model.predict_proba(X_test)
    Y_proba = Y_proba_all[:,1]

    print("\n------ Évaluation du Modèle (sur l'ensemble de test) ------")
    score = recall_score(Y_test, Y_pred) # Sensibilité
    report = classification_report(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    tn, fp, fn, tp = cm.ravel()
    print(f"\nDétail TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
        print(f"\nSpécificité (Taux Vrais Négatifs): {specificity:.4f}")
    else:
        print("\nSpécificité : Non calculable (aucun exemple négatif réel dans le test)")


    print(f"\nSensibilité (Recall): {score:.4f}")
    print("\nClassification Report:")
    print(report)


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

    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_proba)

    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Afficher la courbe ROC
    RocCurveDisplay.from_predictions(Y_test, Y_proba)
    plt.title("Courbe ROC (Logistic Regression - Ensemble de Test)")
    plt.xlabel("Taux de Faux Positifs (1 - Spécificité)")
    plt.ylabel("Taux de Vrais Positifs (Sensibilité)")
    plt.grid(True)
    plt.show()



    # ----------------- TEST AVEC AGGREGATION PAR ID ------------------------

    df_test = pd.DataFrame({
        'id_cas' : ids_test,
        'Y_true': Y_test,
        'Y_pred_emb': Y_pred,
        'Y_proba_0_emb': Y_proba_all[:, 0],  # Probabilité de classe 0
        'Y_proba_1_emb': Y_proba_all[:, 1]  # Probabilité de classe 1 (positive)
    })

    # --- Aggrégation par MOYENNE ----------- Moyenne : 'mean', Mediane : 'median'
    agg_results = df_test.groupby('id_cas').agg(
        Y_true_agg=('Y_true', 'first'),
        Y_proba_0_agg=('Y_proba_0_emb', 'median'),
        Y_proba_1_agg=('Y_proba_1_emb', 'median')
    ).reset_index()


    agg_results['Y_pred_agg'] = (agg_results['Y_proba_1_agg'] >= 0.5).astype(int)

    Y_true_aggregated = agg_results['Y_true_agg'].values
    Y_pred_aggregated = agg_results['Y_pred_agg'].values
    Y_proba_aggregated_pos = agg_results['Y_proba_1_agg'].values

    print(f"****** Évaluation sur {len(agg_results)} {'id_cas'} uniques dans le test set **********")

    score_agg = recall_score(Y_true_aggregated, Y_pred_aggregated)
    report_agg = classification_report(Y_true_aggregated, Y_pred_aggregated)
    cm_agg = confusion_matrix(Y_true_aggregated, Y_pred_aggregated)

    print(f"\nSensibilité (Recall) agrégée par {'id_cas'}: {score_agg:.4f}")
    print(f"\nClassification Report agrégé par {'id_cas'}:")
    print(report_agg)


    print(cm_agg)

    tn_agg, fp_agg, fn_agg, tp_agg = cm_agg.ravel()
    specificity_agg = tn_agg / (tn_agg + fp_agg)
    print(f"\nSpécificité agrégée par {'id_cas'}: {specificity_agg:.4f}")

    print(f"Détail agrégé TN={tn_agg}, FP={fp_agg}, FN={fn_agg}, TP={tp_agg}")

    f1_agg = f1_score(Y_true_aggregated, Y_pred_aggregated)
    roc_auc_agg = roc_auc_score(Y_true_aggregated, Y_proba_aggregated_pos)
    print(f"F1-Score agrégé par {'id_cas'}: {f1_agg:.4f}")
    print(f"ROC AUC agrégé par {'id_cas'}: {roc_auc_agg:.4f}")

    # Plot ROC Curve agrégée
    RocCurveDisplay.from_predictions(Y_true_aggregated, Y_proba_aggregated_pos, name=f"SVM agrégé par {'id_cas'}")
    plt.title(f"Courbe ROC Agrégée par {'id_cas'} (Logistic Regression)")
    plt.grid(True)
    plt.show()


    return X, Y, log_reg_model


def all_probabilities(json_file, log_reg_model, X, pipeline):
    output_dir = "LR_probabilities"
    os.makedirs(output_dir, exist_ok=True)


    with open(json_file) as f:
        data = json.load(f)

    # --- Prédire les probabilités pour TOUTES les données ---
    print("\n------ Prédiction des probabilités pour toutes les données originales ------")
    all_probabilities = log_reg_model.predict_proba(X)

    results_with_probabilities = []
    for i, original_entry in enumerate(data):
        prob_class_0 = all_probabilities[i, 0]
        prob_class_1 = all_probabilities[i, 1]

        if pipeline == "camemBERT":
            result_entry = {
                "id_cas": original_entry["id_cas"],
                "batch": original_entry["batch"],
                "probability_0": prob_class_0,
                "probability_1": prob_class_1
            }
            output_filename_csv = "LR_probabilites_BERT.csv"

        if pipeline == "fastText":
            result_entry = {
                "id_cas": original_entry["id_cas"],
                "probability_0": prob_class_0,
                "probability_1": prob_class_1
            }
            output_filename_csv = "LR_probabilites_fastText.csv"

        results_with_probabilities.append(result_entry)

    print(f"Prédictions de probabilités générées pour {len(results_with_probabilities)} entrées.")


    csv_fieldnames = ["id_cas", "batch", "probability_0", "probability_1"]

    output_path = os.path.join(output_dir, output_filename_csv)

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(results_with_probabilities)

    print(f"Résultats avec probabilités enregistrés avec succès dans '{output_filename_csv}'")

if __name__ == "__main__":
    data_file = 'json_camembert/camembert_sans_meta.json'
    data_file1 = 'data.json'
    pipeline = "camemBERT"
    X, Y, svm_model = classifier_training_lr(data_file)
    all_probabilities(data_file, svm_model, X, pipeline)