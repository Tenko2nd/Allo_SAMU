import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, TFAutoModel

import tensorflow as tf
from typing import List, Tuple
import numpy as np
import json
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

def get_embeddings_from_batches(batched_input_ids: List[List[int]], batched_attention_masks: List[List[int]], model_name: str):
    """
    Génère les embeddings CamemBERT pour une liste de batches tokenisés.
    """
    model = TFAutoModel.from_pretrained(model_name)
    batch_embeddings = []
    for input_ids_batch, attention_mask_batch in zip(batched_input_ids, batched_attention_masks):
        input_ids_tensor = tf.constant([input_ids_batch])
        attention_mask_tensor = tf.constant([attention_mask_batch])
        outputs = model(input_ids_tensor, attention_mask=attention_mask_tensor)
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :].numpy()
        batch_embeddings.append(cls_embedding)
    return batch_embeddings


if __name__ == "__main__":
    data_bert_dir = "../data_bert"
    json_output = []
    use_gpu()

    file_list = [f for f in os.listdir(data_bert_dir) if f.endswith(".txt")]

    for file_name in tqdm(file_list, desc="Processing files", colour="green"):
        file_path = os.path.join(data_bert_dir, file_name)

        # Tokenize text for the current file
        batched_input_ids, batched_attention_masks = tokenize_text(file_path, c.MODEL_NAME, verification=False)

        if batched_input_ids is not None:
            # Get embeddings for the current file
            embeddings_list = get_embeddings_from_batches(batched_input_ids, batched_attention_masks, c.MODEL_NAME)

            # Extract ID_Patient
            ID_Patient = file_name.split("_")[0]

            # Calculate Target
            if ID_Patient[0] in ['A', 'C']:
                Target = 1
            elif ID_Patient[0] in ['B', 'D']:
                Target = 0
            else:
                Target = None
                print(f"Warning: Unknown ID_Patient prefix: {ID_Patient[0]}. Target set to None.")

            for i, embeddings in enumerate(embeddings_list):
                embedding_list = embeddings.tolist()

                batch_json = {
                    "id_cas": ID_Patient,
                    "batch": i + 1,
                    "target": Target,
                    "embedding": embedding_list
                }
                json_output.append(batch_json)

        else:
            print(f"La tokenisation a échoué pour le fichier: {file_name}")

    # Example of saving combined JSON to a file
    output_json_file = "camembert_sans_meta.json"
    with open(output_json_file, 'w') as f:
        json.dump(json_output, f)
    print(f"\nCombined JSON output saved to {output_json_file}")