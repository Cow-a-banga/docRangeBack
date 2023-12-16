from onto import Onto
import nltk
from nltk.tokenize import word_tokenize


def get_phrase_in_sentence(sentence, onto_phrase, morph):
    result = []
    words = word_tokenize(sentence, language="russian")
    morphed_words = [morph.parse(word)[0].normal_form for word in words]
    for phrase_i in range(len(onto_phrase)):
        phrase_words = onto_phrase[phrase_i]
        if len(words) < len(phrase_words):
            continue
        for i in range(len(words) - len(phrase_words) + 1):
            success = True
            for j in range(len(phrase_words)):
                if morphed_words[i + j] != phrase_words[j] and words[i + j] != phrase_words[j]:
                    success = False
                    break
            if success:
                start, length = calculate_mark_indexes(sentence, i, len(phrase_words))
                result.append([phrase_i, start, length])
    return result


def calculate_mark_indexes(line: str, word_index: int, words_num: int):
    counter = -1
    is_word = False
    on_word = False
    start_index = 0
    whitespaces = [' ', '\t', '\n', '\r']
    new_word_symbols = [',', '[', ']', '(', ')', ":"]
    for index, s in enumerate(line):
        if on_word:
            if counter == word_index + words_num - 1 and (s in whitespaces or s in new_word_symbols):
                return start_index, index - start_index

        if s in new_word_symbols:
            counter+=1
            is_word = False
            continue
        if s in whitespaces:
            is_word = False
            continue
        if not is_word:
            counter+=1
            is_word = True
            if counter == word_index:
                start_index = index
                on_word = True
    return start_index, len(line) - start_index - 1


def prepare_positions(positions):
    sorted_positions = sorted([[num[1], num[2]] for num in positions], key=lambda x: (x[0], x[1]))
    result = []
    for index, pos in enumerate(sorted_positions):
        if index + 1 >= len(sorted_positions) or sorted_positions[index+1][0] != pos[0]:
            result.append(pos)
    return result


def get_sentences(text, onto_words_normalized, onto_words, morph):
    sentences = nltk.sent_tokenize(text, language="russian")
    filtered_sentences = list(map(lambda s: [get_phrase_in_sentence(s.lower(), onto_words_normalized, morph), s],sentences))
    return [{"concepts": [onto_words[num[0]] for num in s[0]], "concept_positions":prepare_positions(s[0]), "text": s[1]} for s in filtered_sentences]


def scribe_documents(onto: Onto, morph, documents, useBase):
    scribed_documents = []

    onto_stop_words = [] if useBase else ["Задача", "Аппаратное средство", "Программное средство", "Ресурс", "Метод"]
    onto_words = list(filter(lambda x: x not in onto_stop_words, map(lambda x: x["name"], onto.nodes())))
    onto_words_normalized = [[morph.parse(s.lower())[0].normal_form for s in word.split()] for word in onto_words]

    for document in documents:
        sentences = get_sentences(document["text"], onto_words_normalized, onto_words, morph)
        concepts = set()
        for s in sentences:
            for concept in s["concepts"]:
                concepts.add(concept)
        scribed_documents.append({"name": document["name"], "concepts": list(concepts), "sentences": sentences})
    return scribed_documents
