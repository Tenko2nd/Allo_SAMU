import numpy as np
import joblib
import json

# POUR LA PREDICTION D'UN NOUVEAU CAS (DANS UN .JSON)
def predict_new_case(model_path, new_case_json, threshold=0.5):
    with open(new_case_json) as f:
        data = json.load(f)

    model = joblib.load(model_path)

    X_new = np.array([entry["embedding"] for entry in data])

    probas = model.predict_proba(X_new)[:, 1]

    proba_agg = np.median(probas)

    # Prédiction finale binaire
    pred_label = int(proba_agg >= threshold)

    print(f"Prédiction pour l'id_cas = {data[0]['id_cas']}")
    print(f"Probabilités individuelles : {np.round(probas, 3)}")
    print(f"Probabilité agrégée : {proba_agg:.4f}")
    print(f"Prédiction finale : {pred_label} ({'positif' if pred_label == 1 else 'négatif'})")

    return proba_agg, pred_label


if __name__ == "__main__":
    data_file = 'test.json'
    model_path = "SVM/SVM_model.joblib"
    predict_new_case(model_path, data_file)
