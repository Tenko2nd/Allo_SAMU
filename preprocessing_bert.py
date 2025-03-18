import os
import docx
import spacy


def docx_to_list(record_ID, record_dir):
    doc = docx.Document(record_dir + record_ID)

    # met le texte du document dans une liste
    allText = [docpara.text for docpara in doc.paragraphs]
    allText[0] = " ".join(allText[0].split('\n')[3:])

    cleaned_allText = []
    for paragraph_text in allText:
        cleaned_text = paragraph_text.replace("1-", "")
        cleaned_text = cleaned_text.replace('\n', ' ')
        cleaned_allText.append(cleaned_text)
    allText = cleaned_allText


    return allText


def final_words(all_text, nlp, filename):
    cleaned_text = []
    data_path = "data_bert"
    SEPARATEUR = "/"

    os.makedirs(data_path, exist_ok=True)

    for paragraph in all_text:
        paragraph = paragraph.lower()

        words = paragraph.split()
        filtered_words_paragraph = ' '.join(words)

        cleaned_paragraph = nlp(filtered_words_paragraph)
        for token in cleaned_paragraph:
            cleaned_text.append(token.text)

    filename_txt = filename.replace(".docx", ".txt")
    filename_txt = os.path.join(data_path, filename_txt)


    with open(filename_txt, 'w', encoding='utf-8') as outfile:
        is_first_word = True
        for word in cleaned_text:
            if "speaker" in word and not is_first_word:
                outfile.write(SEPARATEUR+"\n")
            outfile.write(word + " ")
            is_first_word = False

    return cleaned_text

if __name__ == "__main__":
    record_dir = r'C:\Users\maeva\Document\ESEO\E4\S2\Projet_synthese\record/'

    nlp = spacy.load("fr_core_news_lg")

    filenames = os.listdir(record_dir)

    for filename in filenames:
        if filename.endswith(".docx"):
            record_ID = filename

            all_text = docx_to_list(record_ID, record_dir)
            final_word_list = final_words(all_text, nlp, filename)

            print(final_word_list)
            print("-" * 50)