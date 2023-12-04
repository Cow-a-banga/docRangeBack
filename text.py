from onto import Onto
import nltk
from nltk.tokenize import word_tokenize


def get_phrase_in_sentence(sentence, onto_phrase, morph):
    result = []
    words = word_tokenize(sentence)
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
                result.append(phrase_i)
    return result


def get_sentences(text, onto_words_normalized, onto_words, morph):
    sentences = nltk.sent_tokenize(text, language="russian")
    filtered_sentences = list(filter(lambda x: len(x[0]) != 0, map(lambda s: [get_phrase_in_sentence(s.lower(), onto_words_normalized, morph), s],sentences)))
    return [{"concepts": [onto_words[i] for i in s[0]], "text": s[1]} for s in filtered_sentences]


def scribe_documents(onto: Onto, morph, documents):
    scribed_documents = []

    onto_stop_words = ["Задача", "Аппаратное средство", "Программное средство", "Ресурс", "Метод"]
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
