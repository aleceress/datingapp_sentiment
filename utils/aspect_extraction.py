import pandas as pd
import gc
from tqdm.notebook import tqdm
import os
import pickle
from utils import general

def split_sentences(doc):
    sentences = []
    for sent in doc.sents:
        sentences.append(sent)
    return sentences

def find_aspects(sentence):
    aspects = []
    for token in sentence:
        if token.pos_ == "NOUN" and token.text.isalpha():
            adjs = []
            found = False
            for token2 in sentence:
                if token2.pos_ == "ADJ" or token2.lemma_ == "not" and token2.head == token.head:
                    found = True
                    adjs.append(token2.text)
            if found == True:
                aspect = (token.text, adjs)
                aspects.append(aspect)
    return aspects

def explore(token, children=None, level=0, order=None):
    if children is None:
        children = []
    if order is None:
        order = token.idx
    for child in token.children:
        children.append((child, level, child.idx < order))
        explore(child, children=children, level=level+1, order=order)
    return children

def add_adj_aspects(nouns_map, nouns_freq, nlp_text, nouns=None):
    if nouns is None:
        nouns = [x for x in nlp_text if x.pos_ in ['NOUN', 'PROPN'] and x.text.isalpha()]

    for noun in nouns:
        subtree = explore(noun)
        subnouns = [(x, l)
                    for x, l, _ in subtree if x.pos_ in ['NOUN', 'PROPN']]
        for token, level, _ in subtree:
            if token.pos_ == 'ADJ' and len([(n, l) for n, l in subnouns if l < level]) == 0:
                try:
                    nouns_map[noun.text.lower()].add(token.text.lower())
                    nouns_freq[noun.text.lower()]+=1
                except KeyError:
                    nouns_map[noun.text.lower()] = {token.text.lower()}
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
                        try:
                            nouns_map[subject.text.lower()].add(candidate.text.lower())
                            nouns_freq[subject.text.lower()]+=1
                        except KeyError:
                            nouns_map[subject.text.lower()] = {candidate.text.lower()}
                            nouns_freq[subject.text.lower()] = 1
                    elif candidate.dep_ in ['dobj', 'attr', 'conj'] and candidate.text.isalpha():
                        add_adj_aspects(nouns_map, nouns_freq, nlp_text, nouns=[candidate])
              

def get_aspects_adjs_and_freq():
    if os.path.exists("data/aspect_adjs.pickle") and os.path.exists("data/aspect_freq.pickle"):
        with open("data/aspect_adjs.pickle", "rb") as f:
            aspect_adjs = pickle.load(f)
        return aspect_adjs, pd.read_pickle("data/aspect_freq.pickle")
    
    aspect_adjs = {}
    aspect_freq = pd.Series()

    for i in range(10):
        print(f"processing {i+1}/10 reviews")
        gc.collect()
        tinder_reviews = pd.read_pickle(f"data/tinder_spacy/spacy_tinder_sample_{i}.pickle").dropna()
        for _, review in tqdm(tinder_reviews["content"].items(), total=len(tinder_reviews)):
            for _, sentence in enumerate(split_sentences(review)):
                add_adj_aspects(aspect_adjs, aspect_freq, sentence)
                add_verb_aspects(aspect_adjs, aspect_freq, sentence)
    
    norm_aspect_freq = general.normalize_series(aspect_freq)
    norm_aspect_freq.to_pickle("data/aspect_freq.pickle")

    with open("data/aspect_adjs.pickle", "wb+") as f:
        pickle.dump(aspect_adjs, f)
        
    return aspect_adjs, norm_aspect_freq