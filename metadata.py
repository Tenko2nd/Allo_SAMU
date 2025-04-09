import os
import pandas as pd

# === 1. Lire le fichier Excel avec les métadonnées ===
excel_path = "C:/Users/diopndey/Documents/projet de synthese/Data/Cas-AnonymeFINAL.xlsx"  # À adapter si besoin
df = pd.read_excel(excel_path)


# === 2. Fonctions de normalisation ===
def normalize_sex(value):
    if value == 'M':
        return 0
    elif value == 'F':
        return 1
    return 0.5  # Inconnu


def normalize_age(age):
    if pd.isna(age):
        return None
    return max(0, min(1, (age - 30) / (120 - 30)))


def normalize_month(date):
    if pd.isna(date):
        return None
    if isinstance(date, str):
        date = pd.to_datetime(date, errors='coerce')
    if pd.isna(date):
        return None
    return (date.month - 1) / 11


# === 3. Appliquer la normalisation ===
df["sexe_normalized"] = df["Sexe"].apply(normalize_sex)
df["age_normalized"] = df["Age"].apply(normalize_age)
df["mois_normalized"] = df["Date Appel"].apply(normalize_month)

# === 4. Mapping id_idm -> valeurs normalisées ===
df["id_idm"] = df["id_idm"].astype(str)  # On force en string
df = df.drop_duplicates(subset="id_idm", keep="first")
metadata_dict = df.set_index("id_idm")[["age_normalized", "sexe_normalized", "mois_normalized"]].to_dict("index")

# === 5. Charger les fichiers du dossier "data" ===
data_dir = "data"
files = [f for f in os.listdir(data_dir) if isinstance(f, str) and f.endswith(".txt")]

output_path = "output/metadonnees_normalisees.txt"

# === 6. Associer chaque document à ses métadonnées ===

with open(output_path, "w", encoding="utf-8") as out_file:
    for file in files:
        try:
            file_lower = str(file).lower()  # S'assurer que c'est bien une string
            for id_idm, metadata in metadata_dict.items():
                if str(id_idm).lower() in file_lower:
                    ligne = f"{file}: [âge: {metadata['age_normalized']}, sexe: {metadata['sexe_normalized']}, mois: {metadata['mois_normalized']}]\n"
                    out_file.write(ligne)
                    break
        except Exception as e:
            out_file.write(f"Erreur avec le fichier {file} : {e}\n")