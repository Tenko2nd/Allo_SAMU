import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, TFAutoModel

import tensorflow as tf
from typing import List, Tuple
import numpy as np
import json
import csv
from tqdm import tqdm # Import tqdm

import camembert_constant as c
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

def get_embeddings_from_batches(batched_input_ids: List[List[int]], batched_attention_masks: List[List[int]], model):
    """
    Génère les embeddings BERT pour une liste de batches tokenisés.
    """
    batch_embeddings = []
    for input_ids_batch, attention_mask_batch in zip(batched_input_ids, batched_attention_masks):
        input_ids_tensor = tf.constant([input_ids_batch])
        attention_mask_tensor = tf.constant([attention_mask_batch])
        outputs = model(input_ids_tensor, attention_mask=attention_mask_tensor)
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :].numpy()
        batch_embeddings.append(cls_embedding)
    return batch_embeddings


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

        if batched_input_ids is not None:
            embeddings_list = get_embeddings_from_batches(batched_input_ids, batched_attention_masks, model)

            ID_Patient = file_name.split("_")[0]

            Target = 1 if ID_Patient[0] in ['A', 'C'] else 0 if ID_Patient[0] in ['B', 'D'] else None
            if Target is None:
                print(f"Warning: Unknown ID_Patient prefix: {ID_Patient[0]}. Target set to None.")

            if type_metadata is not None:
                normalized_metadata = get_normalized_metadata(ID_Patient, metadata_dict, type_metadata)

            for i, embeddings in enumerate(embeddings_list):
                embeddings = embeddings.flatten()
                if type_metadata is not None:
                    embeddings = np.concatenate((embeddings, normalized_metadata)).tolist()

                batch_json = {
                    "id_cas": ID_Patient,
                    "batch": i + 1,
                    "target": Target,
                    "embedding": embeddings.tolist()
                }
                json_output.append(batch_json)
        else:
            print(f"La tokenisation a échoué pour le fichier: {file_name}")
    return json_output


if __name__ == "__main__":
    data_bert_dir = "data_bert_w_nlp"
    csv_file_path = "../Cas-AnonymeFINAL.csv"
    list_type_metadata = [None]

    use_gpu()

    metadata_dict = load_metadata_csv(csv_file_path)

    file_list = [f for f in os.listdir(data_bert_dir) if f.endswith(".txt")]

    tokenizer = AutoTokenizer.from_pretrained(c.MODEL_NAME)
    model = TFAutoModel.from_pretrained(c.MODEL_NAME, from_pt=True)

    for type_metadata in list_type_metadata:
        print(f"\nProcessing type {type_metadata} for model {c.MODEL_NAME}")
        json_output = embeddings_to_json(data_bert_dir=data_bert_dir, file_list=file_list, tokenizer=tokenizer, model=model, type_metadata=None)

        output_dir = f"{c.MODEL_NAME.split('/')[-1]}_json"
        json_file = f"{c.MODEL_NAME.split('/')[-1]}_{type_metadata if type_metadata is not None else 'sans_metadata'}_{c.CONTEXT_LEN}.json"

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            print(f"\nCreated directory {output_dir}")

        with open(f"{output_dir}/{json_file}", 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"\nCombined JSON output saved to {output_dir}/{json_file}")