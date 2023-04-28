import pandas as pd
import gc
from tqdm.notebook import tqdm
import os
import pickle
from utils import general, sentiment_extraction
from nltk.corpus import wordnet as wn
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


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

def get_aspects_adjs_and_freq(app = "tinder"):

    if os.path.exists(f"data/aspects_adjs_{app}.pickle") and os.path.exists(f"data/aspects_freq_{app}.pickle"):
        with open(f"data/aspects_adjs_{app}.pickle", "rb") as f:
            aspects_adjs = pickle.load(f)
        return aspects_adjs, pd.read_pickle(f"data/aspects_freq_{app}.pickle")
   
    aspects_adjs = {}
    aspects_freq = pd.Series()

    review_files = os.listdir(f"data/{app}_spacy")
    for i, filename in enumerate(review_files):
        print(f"processing {i+1}/{len(review_files)} reviews")
        gc.collect()
        reviews = pd.read_pickle(f"data/{app}_spacy/{filename}").dropna()
        for _, review in tqdm(reviews["content"].items(), total=len(reviews)):
            for _, sentence in enumerate(split_sentences(review)):
                add_adj_aspects(aspects_adjs, aspects_freq, sentence)
                add_verb_aspects(aspects_adjs, aspects_freq, sentence)
    
    norm_aspects_freq = general.normalize_series(aspects_freq)
    norm_aspects_freq.to_pickle(f"data/aspects_freq_{app}.pickle")

    with open(f"data/aspects_adjs_{app}.pickle", "wb+") as f:
        pickle.dump(aspects_adjs, f)
        
    return aspects_adjs, norm_aspects_freq

def cluster_query_adjs(query_aspects, aspects_adjs, polarity= "all"):
    adjs = []
    for aspect in query_aspects:
        for adj in aspects_adjs[aspect]:
            if polarity == "pos" and sentiment_extraction.get_sentiwn_score(adj) > 0.2:
                adjs.append(adj)
            elif polarity == "neg" and sentiment_extraction.get_sentiwn_score(adj) < - 0.2:
                adjs.append(adj)
            elif polarity == "all":
                adjs.append(adj)
            

    G = nx.Graph()
    min_sim = 0.6
    for word1 in tqdm(adjs):
        for word2 in adjs:
            if word1 != word2:
                try:
                    syn1 = wn.synsets(word1)[0]
                    syn2 = wn.synsets(word2)[0]
                    sim = syn1.wup_similarity(syn2)
                    if  sim > min_sim:
                        G.add_edge(word1, word2, sim=sim)
                except:
                    continue

    if G.number_of_nodes() != 0:
        communities = greedy_modularity_communities(G)
        for community in communities:
            print(list(community))
    else:
        communities = None 
    return communities, G
