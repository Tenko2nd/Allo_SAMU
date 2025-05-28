import os
import json

# === 0. Lire les métadonnées d'âge depuis un fichier .txt ===
metadata_path = "infos_patients_agee_normalisees.txt"  #  adapte ici le nom si nécessaire
metadata = {}

with open(metadata_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("→")
        if len(parts) == 2:
            id_cas = parts[0].strip().lower()
            try:
                age = float(parts[1].split("=")[1].strip())
            except:
                age = 0.0  # par défaut si valeur incorrecte
            metadata[id_cas] = age
        else:
            metadata[parts[0].strip().lower()] = 0.0  # si format incorrect

# === 1. Lire les fichiers TF-IDF (.txt) ===
concat_results = []
tfidf_folder = "tfidf_results"

for file in os.listdir(tfidf_folder):
    if file.endswith("_tfidf.txt"):
        id_cas = file[:3].lower().replace("_", "").replace("-", "")
        tfidf_path = os.path.join(tfidf_folder, file)

        with open(tfidf_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            tfidf_vector = []
            for line in lines[3:]:  
                words = line.strip().split()
                for w in words:
                    try:
                        tfidf_vector.append(float(w))
                    except:
                        pass

        # Déterminer la target
        first_char = id_cas[0]
        target = 1 if first_char in {"a", "c", "w", "y"} else 0 if first_char in {"b", "d", "x", "z"} else None

        # Ajouter métadonnées au début du vecteur
        age = metadata.get(id_cas, 0.0)  # à adapter en fonction de la métadonnées à intégrer
        embedding = [age] + tfidf_vector

        concat_results.append({
            "id_cas": id_cas,
            "target": target,
            "embedding": embedding
        })

# === 2. Export en JSON
output_path = "json_avec_age.json"


with open(output_path, "w", encoding="utf-8") as out:
    json.dump(concat_results, out, ensure_ascii=False, indent=4)

print(f"✅ Export terminé dans '{output_path}' avec l'âge au début du vecteur embedding")
