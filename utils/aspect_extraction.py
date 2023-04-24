import pandas as pd
import gc
from tqdm.notebook import tqdm
import os
import pickle
from utils import general
from nltk.corpus import wordnet as wn

def split_sentences(doc):
    sentences = []
    for sent in doc.sents:
        sentences.append(sent)
    return sentences


def explore(token, children=None, level=0, order=None):
    if children is None:
        children = []
    if order is None:
        order = token.idx
    for child in token.children:
        children.append((child, level, child.idx < order))
        explore(child, children=children, level=level+1, order=order)
    return children

def get_antonyms_for(word):
    antonyms = set()
    for ss in wn.synsets(word):
        for lemma in ss.lemmas():
            any_pos_antonyms = [ antonym.name() for antonym in lemma.antonyms() ]
            for antonym in any_pos_antonyms:
                antonym_synsets = wn.synsets(antonym)
                if wn.ADJ not in [ ss.pos() for ss in antonym_synsets ]:
                    continue
                antonyms.add(antonym)
    return antonyms

def add_adj_aspects(nouns_map, nouns_freq, nlp_text, nouns=None):
    if nouns is None:
        nouns = [x for x in nlp_text if x.pos_ in ['NOUN', 'PROPN'] and x.text.isalpha()]

    for noun in nouns:
        subtree = explore(noun)

        subnouns = [(x, l)
                    for x, l, _ in subtree if x.pos_ in ['NOUN', 'PROPN']]
        for token, level, _ in subtree:
            if token.pos_ == 'ADJ' and len([(n, l) for n, l in subnouns if l < level]) == 0:
                for child in token.children:
                    if child.lemma_ == "not" and len(get_antonyms_for(str(token))) != 0:
                        token = next(iter(get_antonyms_for(str(token))))
                try:
                    nouns_map[noun.text.lower()].append(str(token).lower())
                    nouns_freq[noun.text.lower()]+=1
                except KeyError:
                    nouns_map[noun.text.lower()] = [str(token).lower()]
                    nouns_freq[noun.text.lower()] = 1

def add_verb_aspects(nouns_map, nouns_freq, nlp_text, be_only=True):
    if be_only:
        verbs = [x for x in nlp_text if x.lemma_ == 'be']
    else:
        verbs = [x for x in nlp_text if x.pos_ in {'AUX', 'VERB'}]
    for verb in verbs:
        subtokens = explore(verb)
        subject = [(x)
                   for x, level, left in subtokens if left and x.dep_ == 'nsubj']
        if len(subject) > 0:
            subject = subject[0]
            for candidate, level, left in subtokens:
                if not left:
                    if candidate.pos_ == 'ADJ' and candidate.text.isalpha() and level == 0:
                        for subtoken in subtokens:
                            if subtoken[0].lemma_ == "not" and len(get_antonyms_for(str(candidate))) != 0:
                                candidate = next(iter(get_antonyms_for(str(candidate))))
                        try:
                            nouns_map[subject.text.lower()].append(str(candidate).lower())
                            nouns_freq[subject.text.lower()]+=1
                        except KeyError:
                            nouns_map[subject.text.lower()] = [str(candidate).lower()]
                            nouns_freq[subject.text.lower()] = 1
                    elif candidate.dep_ in ['dobj', 'attr', 'conj'] and candidate.text.isalpha():
                        add_adj_aspects(nouns_map, nouns_freq, nlp_text, nouns=[candidate])

def get_aspects_adjs_and_freq():
    if os.path.exists("data/aspects_adjs.pickle") and os.path.exists("data/aspects_freq.pickle"):
        with open("data/aspectprint(review)s_adjs.pickle", "rb") as f:
            aspects_adjs = pickle.load(f)
        return aspects_adjs, pd.read_pickle("data/aspects_freq.pickle")
    
    aspects_adjs = {}
    aspects_freq = pd.Series()

    for i in range(10):
        print(f"processing {i+1}/10 reviews")
        gc.collect()
        tinder_reviews = pd.read_pickle(f"data/tinder_spacy/spacy_tinder_sample_{i}.pickle").dropna()
        for _, review in tqdm(tinder_reviews["content"].items(), total=len(tinder_reviews)):
            for _, sentence in enumerate(split_sentences(review)):
                add_adj_aspects(aspects_adjs, aspects_freq, sentence)
                add_verb_aspects(aspects_adjs, aspects_freq, sentence)
    
    norm_aspects_freq = general.normalize_series(aspects_freq)
    norm_aspects_freq.to_pickle("data/aspects_freq.pickle")

    with open("data/aspects_adjs.pickle", "wb+") as f:
        pickle.dump(aspects_adjs, f)
        
    return aspects_adjs, norm_aspects_freq