from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, confusion_matrix
import numpy as np
import json

# ------- Mettre les data avec tous les dictionnaires -----------
# ------- Ouvrir le .json avec tous les dictionnaires -----------

json_file = r"Z:\camembert.json"
with open(json_file) as f:
    data = json.load(f)

X = np.array([entry["embedding"] for entry in data])
Y = np.array([entry["target"] for entry in data])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_classifier.fit(X_train, Y_train)

Y_pred = rf_classifier.predict(X_test)

# ---------- Evaluation ----------------------

score = recall_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
specificity = tn / (tn + fp)

print("------ Évaluation Binaire ------")

print(f"Sensibilité : {score:.4f}")

print("\nClassification Report:")
print(report)

print("\nMatrice de Confusion:")
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

if (tn + fp) > 0:
    specificity = tn / (tn + fp)
    print(f"\nSpecificite (Taux Vrais Négatifs): {specificity:.4f}")
else:
    print("\nSpecificite : Non calculable (aucun exemple négatif réel)")

print(f"\nDétail TN={tn}, FP={fp}, FN={fn}, TP={tp}")
