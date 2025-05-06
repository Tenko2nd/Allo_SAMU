import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, TFAutoModel

import tensorflow as tf
from typing import List, Tuple
import numpy as np
import json
import csv
from tqdm import tqdm # Import tqdm

import bert_constant as c
from tokenization_par_batch import tokenize_text

import warnings

warnings.filterwarnings("ignore")


def use_gpu():
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available and will be used.")
        try:
            tf.config.set_visible_devices(gpus, 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU detected. TensorFlow will use CPU.")

def get_embeddings_for_file_batched(all_input_ids: List[List[int]], all_attention_masks: List[List[int]], model):
    """
    Génère les embeddings BERT pour tous les batches d'un fichier en une seule passe modèle.
    """
    if not all_input_ids: # Vérifier si la liste est vide
        return []

    input_ids_tensor = tf.constant(all_input_ids, dtype=tf.int32)
    attention_mask_tensor = tf.constant(all_attention_masks, dtype=tf.int32)

    # Appel unique au modèle pour tous les batches
    outputs = model(input_ids_tensor, attention_mask=attention_mask_tensor)
    last_hidden_states = outputs.last_hidden_state

    # Extraire les embeddings [CLS] pour chaque élément du batch
    cls_embeddings = last_hidden_states[:, 0, :].numpy() # Shape: (nombre_de_batches, hidden_size)
    return list(cls_embeddings) # Retourne une liste d'arrays numpy, un par batch original


def load_metadata_csv(csv_file_path: str) -> dict[str, dict[str, any]]:
    """
    Loads metadata from a CSV file and returns it as a dictionary.
    Keys are patient IDs, values are dictionaries of metadata fields.
    """
    metadata_dict = {}
    with open(csv_file_path, mode='r', encoding="utf-8-sig") as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=';')
        for row in csv_reader:
            patient_id = row['id_idm'].lower() # Ensure consistent ID format (lowercase)
            metadata_dict[patient_id] = row
    return metadata_dict

def get_normalized_metadata(patient_id: str, metadata: dict[str, dict[str, any]], type_metadata: str) -> np.ndarray:
    """
    Extracts, normalizes, and returns metadata as a NumPy array.
    """
    patient_metadata = metadata.get(patient_id.lower()) # Use lowercase for lookup

    if patient_metadata:
        try:
            # Normalize Age (assuming age range 30-120)
            age = int(patient_metadata['Age'])
            normalized_age = ((age - 30) / (120 - 30)) - 0.

            # Sexe
            sexe = 0.5 if patient_metadata['Sexe'].upper() == 'M' else -0.5

            if type_metadata == "A":
                return np.array([normalized_age], dtype=np.float32)
            if type_metadata == "S":
                return np.array([sexe], dtype=np.float32)
            if type_metadata == "AS":
                return np.array([normalized_age, sexe], dtype=np.float32)
            else:
                print(f"Error mtadata type unknown : {type_metadata}")
                return np.array([0, 0], dtype=np.float32)

        except (ValueError, KeyError) as e:
            print(f"Error processing metadata for patient {patient_id}: {e}")
            return np.array([0, 0], dtype=np.float32) # Default metadata
    else:
        print(f"Warning: No metadata found for patient ID: {patient_id}")
        return np.array([0, 0], dtype=np.float32)


def embeddings_to_json(data_bert_dir, file_list, tokenizer, model, type_metadata=None):
    json_output = []

    for file_name in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(data_bert_dir, file_name)

        batched_input_ids, batched_attention_masks = tokenize_text(file_path, tokenizer, verification=False)

        if batched_input_ids is not None and batched_input_ids:  # Vérifier aussi que ce n'est pas vide
            # Appel à la nouvelle fonction
            embeddings_list_np = get_embeddings_for_file_batched(batched_input_ids, batched_attention_masks, model)

            ID_Patient = file_name.split("_")[0]

            Target = 1 if ID_Patient[0] in ['A', 'C', 'W', 'Y'] else 0 if ID_Patient[0] in ['B', 'D', 'X', 'Z'] else None
            if Target is None:
                print(f"Warning: Unknown ID_Patient prefix: {ID_Patient[0]}. Target set to None.")

            if type_metadata is not None:
                normalized_metadata = get_normalized_metadata(ID_Patient, metadata_dict, type_metadata)

            for i, embeddings_np in enumerate(embeddings_list_np):
                if type_metadata is not None:
                    final_embeddings = np.concatenate((embeddings_np, normalized_metadata))
                else:
                    final_embeddings = embeddings_np

                batch_json = {
                    "id_cas": ID_Patient,
                    "batch": i + 1,
                    "target": Target,
                    "embedding": final_embeddings.tolist()
                }
                json_output.append(batch_json)
        else:
            print(f"La tokenisation a échoué pour le fichier: {file_name}")
    return json_output


if __name__ == "__main__":
    list_data_bert_dir = ["data_bert_nlp", "data_bert_raw"]
    csv_file_path = "../Données_finales.csv"
    list_type_metadata = ["A", "AS", "S", None]
    # ["almanach/camembertav2-base", "flaubert/flaubert_large_cased", "almanach/camembert-base", "Dr-BERT/DrBERT-7GB"] ordre de qualité décroissant après test
    list_model_name = ["flaubert/flaubert_large_cased", "almanach/camembertav2-base"] # Les deux meilleurs
    use_gpu()

    metadata_dict = load_metadata_csv(csv_file_path)

    for data_bert_dir in list_data_bert_dir:

        for model_name in list_model_name:
            file_list = [f for f in os.listdir(data_bert_dir) if f.endswith(".txt")]

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = TFAutoModel.from_pretrained(model_name, from_pt=True)

            for type_metadata in list_type_metadata:
                print(f"\nProcessing type {type_metadata} for model {model_name}")
                json_output = embeddings_to_json(data_bert_dir=data_bert_dir, file_list=file_list, tokenizer=tokenizer, model=model, type_metadata=type_metadata)

                output_dir = f"{model_name.split('/')[-1]}_{('_').join(data_bert_dir.split('_')[-1:])}_json"
                json_file = f"{model_name.split('/')[-1]}_{type_metadata if type_metadata is not None else 'sans_metadata'}_{c.CONTEXT_LEN}.json"

                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                    print(f"\nCreated directory {output_dir}")

                with open(f"{output_dir}/{json_file}", 'w') as f:
                    json.dump(json_output, f, indent=2)
                print(f"\nCombined JSON output saved to {output_dir}/{json_file}")