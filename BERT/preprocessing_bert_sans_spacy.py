import os
import docx
import re
from tqdm import tqdm

import bert_constant as c

# === Extraction du contenu du .docx et insertion des balises SEP ===
def docx_to_list(record_ID, record_dir):
    doc = docx.Document(record_dir + record_ID)

    allText = [docpara.text for docpara in doc.paragraphs]
    if len(allText[0].split('\n')) > 3:
        allText[0] = " ".join(allText[0].split('\n')[3:])

    cleaned_allText = []
    for paragraph_text in allText:
        paragraph_text = paragraph_text.replace("1-", "")
        paragraph_text = paragraph_text.replace('SPEAKER_01', '[SEP]\nDocteur:')
        paragraph_text = paragraph_text.replace('SPEAKER_00', '[SEP]\nInterlocuteur:')
        paragraph_text = paragraph_text.replace('SPEAKER_02', '[SEP]\nInterlocuteur:')
        paragraph_text = paragraph_text.replace('SPEAKER_03', '[SEP]\nInterlocuteur:')
        cleaned_allText.append(paragraph_text)

    if cleaned_allText:
        # Supprime les espaces avant le tout premier [SEP]\n au début du texte
        cleaned_allText[0] = re.sub(r'^\s*\[SEP\]\n', '', cleaned_allText[0], count=1)

    return cleaned_allText

# === Traitement simple sans SpaCy ===
def final_words(all_text, filename):
    cleaned_text = []
    data_path = "data_bert_raw"
    SEPARATEUR = c.TEXT_FILE_SEPARATOR
    spec_char = []

    os.makedirs(data_path, exist_ok=True)

    with open('../Ressources/caracteres_speciaux.txt', 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            for char in line:
                spec_char.append(char)

    for paragraph in all_text:
        words = paragraph.split()
        nouveau_words = []

        for word in words:
            if word == SEPARATEUR:
                nouveau_words.append(word)
                continue

            mot_split = False
            for char in spec_char:
                if len(word) == 2 and char in word:
                    nouveau_words.extend([word[0], word[1]])
                    mot_split = True
            if not mot_split:
                nouveau_words.append(word)

        for i, token in enumerate(nouveau_words):
            if token == SEPARATEUR:
                cleaned_text.append(SEPARATEUR)
            elif token == ":" and cleaned_text and cleaned_text[-1].lower() in ["docteur", "interlocuteur"]:
                cleaned_text.append(":")
            elif token.strip():
                cleaned_text.append(token.strip())

    filename_txt = filename.replace(".docx", ".txt")
    filename_txt = os.path.join(data_path, filename_txt)

    with open(filename_txt, 'w', encoding='utf-8') as outfile:
        is_start_of_line = True

        for i, word in enumerate(cleaned_text):
            if word == SEPARATEUR:
                outfile.write(SEPARATEUR)
                outfile.write("\n")
                is_start_of_line = True
                continue

            if not is_start_of_line:
                outfile.write(" ")

            outfile.write(word)
            is_start_of_line = False

    return cleaned_text


if __name__ == "__main__":
    record_dir = r'C:\Users\casserma\Documents\Data\Retranscriptions Anonymes_FINAL/' # A modifier avec le dossier où se trouvent les enregistrements .docx

    filenames = os.listdir(record_dir)

    for filename in tqdm(filenames):
        if filename.endswith(".docx"):
            record_ID = filename

            all_text = docx_to_list(record_ID, record_dir)
            final_word_list = final_words(all_text, filename)

