import os
import json

# === 1. Lire les fichiers TF-IDF (.txt) ===
concat_results = []
tfidf_folder = "output/tfidf_results"

for file in os.listdir(tfidf_folder):
    if file.endswith("_tfidf.txt"):
        id_cas = file[:3].lower()
        tfidf_path = os.path.join(tfidf_folder, file)

        with open(tfidf_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            tfidf_vector = []
            for line in lines[3:]:  # sauter les 3 premières lignes descriptives
                words = line.strip().split()
                for w in words:
                    try:
                        tfidf_vector.append(float(w))
                    except:
                        pass  # ignorer les non-nombres

                # Détermination de la target
                first_char = id_cas[0]
                target = 1 if first_char in {"a", "c"} else 0 if first_char in {"b", "d"} else None

        concat_results.append({
            "id_cas": id_cas,
            "target": target,
            "embedding": tfidf_vector
        })

# === 2. Export en JSON
with open("output/json_sansmetadata.json", "w", encoding="utf-8") as out:
    json.dump(concat_results, out, ensure_ascii=False, indent=4)

print("✅ Export TF-IDF uniquement terminé dans 'json_tfidf_only.json'")
