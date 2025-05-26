import json
from collections import OrderedDict

json_file = r"Z:\Allo_SAMU\classifier\json_fasttext\new\json_sansmetadata.json"

# Charger le fichier JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Récupérer les 204 derniers id_cas uniques (en partant de la fin)
seen_ids = OrderedDict()
for item in (data):
    id_cas = item["id_cas"]
    if id_cas not in seen_ids:
        seen_ids[id_cas] = True
    if len(seen_ids) == 204:
        break

# Extraire tous les éléments correspondant aux id_cas sélectionnés
selected_id_cas = set(seen_ids.keys())
filtered_data = [item for item in data if item["id_cas"] in selected_id_cas]

# Sauvegarder le résultat dans un nouveau fichier JSON
with open("classifier/json_bert/json_new_2/camembert_new/camembert_test.json", "w", encoding="utf-8") as f_out:
    json.dump(filtered_data, f_out, ensure_ascii=False, indent=2)

# Afficher les compteurs
print(f"Nombre d'id_cas uniques conservés : {len(selected_id_cas)}")
print(f"Nombre total de dictionnaires dans la nouvelle liste : {len(filtered_data)}")
print("Les données ont été sauvegardées dans 'fichier_204_derniers.json'")
