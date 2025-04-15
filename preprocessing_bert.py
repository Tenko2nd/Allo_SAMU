import os
import docx
import spacy
from spacy.symbols import ORTH # <--- AJOUTER CECI
import re

def docx_to_list(record_ID, record_dir, option):
    doc = docx.Document(record_dir + record_ID)

    allText = [docpara.text for docpara in doc.paragraphs]
    if len(allText[0].split('\n')) > 3:
        allText[0] = " ".join(allText[0].split('\n')[3:])

    cleaned_allText = []
    for paragraph_text in allText:
        paragraph_text = paragraph_text.replace("1-", "")

        if option == "avec_speaker":
            paragraph_text = paragraph_text.replace('SPEAKER_01', '[SEP]\nDocteur:')
            paragraph_text = paragraph_text.replace('SPEAKER_00', '[SEP]\nInterlocuteur:')
            paragraph_text = paragraph_text.replace('SPEAKER_02', '[SEP]\nInterlocuteur:')
            paragraph_text = paragraph_text.replace('SPEAKER_03', '[SEP]\nInterlocuteur:')
        elif option == "sans_speaker":
            paragraph_text = paragraph_text.replace('SPEAKER_01', '[SEP]\n')
            paragraph_text = paragraph_text.replace('SPEAKER_00', '[SEP]\n')
            paragraph_text = paragraph_text.replace('SPEAKER_02', '[SEP]\n')
            paragraph_text = paragraph_text.replace('SPEAKER_03', '[SEP]\n')
        cleaned_allText.append(paragraph_text)

    if cleaned_allText:
        # Supprime les espaces avant le tout premier [SEP]\n au début du texte
        cleaned_allText[0] = re.sub(r'^\s*\[SEP\]\n', '', cleaned_allText[0], count=1)

    return cleaned_allText


def final_words(all_text, nlp, filename, option):
    cleaned_text = []
    data_path = "data_bert"
    SEPARATEUR = "[SEP]"
    spec_char = []

    os.makedirs(data_path, exist_ok=True)

    with open('Ressources/caracteres_speciaux.txt', 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            for char in line:
                spec_char.append(char)

    # Traitement des paragraphes et tokenisation (inchangé)
    for paragraph in all_text:
        paragraph = paragraph.replace(SEPARATEUR, "<<SEP>>")
        paragraph = paragraph.lower()
        paragraph = paragraph.replace("<<sep>>", SEPARATEUR)

        words = paragraph.split()
        nouveau_words = []

        for word in words:
            mot_split = False
            for char in spec_char:
                if len(word) == 2 and char in word:
                    nouveau_words.extend([word[0], word[1]])
                    mot_split = True
                elif char in word and char not in [".", ",", ";", "?", "!", "[","]"]:
                    parts = word.split(char, 1)
                    nouveau_words.extend([parts[0], char, parts[1]])
                    mot_split = True
                    break
            if not mot_split:
                nouveau_words.append(word)


        print(nouveau_words)
        # if option == "sans_speaker":
        #     if len(nouveau_words) >= 2:
        #         del nouveau_words[0]
        #         del nouveau_words[0]
        #     i = 0
        #     while i < len(nouveau_words):
        #         if nouveau_words[i] == SEPARATEUR:
        #             # Supprimer les deux mots suivants si possibles
        #             if i + 1 < len(nouveau_words):
        #                 del nouveau_words[i + 1]
        #             if i + 1 < len(nouveau_words):  # encore une fois (la liste a déjà changé)
        #                 del nouveau_words[i + 1]
        #         i += 1  # on avance normalement
        #     print("-------------------------------- \n")
        #     print(nouveau_words)

        filtered_words_paragraph = ' '.join(nouveau_words)
        cleaned_paragraph = nlp(filtered_words_paragraph)

        for token in cleaned_paragraph:
            if token.text == SEPARATEUR:
                 cleaned_text.append(SEPARATEUR)
            elif token.text == ":" and cleaned_text and cleaned_text[-1].lower() in ["docteur", "interlocuteur"]:
                 cleaned_text.append(token.text)
            # Exclure les tokens vides ou juste des espaces (si SpaCy en produisait)
            elif token.text.strip():
                 cleaned_text.append(token.text.strip())


    filename_txt = filename.replace(".docx", ".txt")
    filename_txt = os.path.join(data_path, filename_txt)

    with open(filename_txt, 'w', encoding='utf-8') as outfile:
        is_start_of_line = True

        for i, word in enumerate(cleaned_text):
            if word == SEPARATEUR:

                outfile.write(SEPARATEUR)
                outfile.write("\n")
                is_start_of_line = True #
                continue

            # if option == "sans_speaker" and word.lower() in ["docteur", "interlocuteur", ":"]:
            #      # Si le mot filtré est suivi de :, on le saute aussi potentiellement
            #      continue

            if not is_start_of_line:
                outfile.write(" ")

            outfile.write(word)
            is_start_of_line = False

    return cleaned_text


if __name__ == "__main__":
    record_dir = r'C:\Users\ramamoma\Documents\data/' # A modifier avec le dossier où se trouvent les enregistrements .docx

    nlp = spacy.load("fr_core_news_lg")

    # Règle spéciale pour que "[SEP]" soit toujours traité comme un seul token
    special_case = [{ORTH: "[SEP]"}]
    nlp.tokenizer.add_special_case("[SEP]", special_case)

    filenames = os.listdir(record_dir)
    option = "sans_speaker"

    for filename in filenames:
        if filename.endswith(".docx"):
            record_ID = filename

            all_text = docx_to_list(record_ID, record_dir,option)
            final_word_list = final_words(all_text, nlp, filename, option)
            #
            # print(final_word_list)
            # print("-" * 50)
