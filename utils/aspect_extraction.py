import pandas as pd
import gc
from tqdm.notebook import tqdm
import os
import pickle
from utils import general, sentiment_extraction
from nltk.corpus import wordnet as wn
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import collections
import nltk
import numpy as np
import itertools

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

def get_aspects_adjs_and_freq(app = "tinder", evaluation_ids = None):
    
    if os.path.exists(f"data/aspects_adjs_{app}.pickle") and os.path.exists(f"data/aspects_freq_{app}.pickle") and evaluation_ids is None:
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
        if evaluation_ids is not None:
            reviews = reviews[reviews.reviewId.isin(evaluation_ids)]
        for _, review in tqdm(reviews["content"].items(), total=len(reviews)):
            for _, sentence in enumerate(split_sentences(review)):
                add_adj_aspects(aspects_adjs, aspects_freq, sentence)
                add_verb_aspects(aspects_adjs, aspects_freq, sentence)
    
    norm_aspects_freq = general.normalize_series(aspects_freq)
    norm_aspects_freq.to_pickle(f"data/aspects_freq_{app}.pickle")

    if evaluation_ids is None:
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

def get_adj_noun_pmi(reviews):
    words = []
    for review in reviews:
        words.extend(review)

    word_frequencies = collections.Counter(words)
    word_frequencies = {word[0]: word[1] for word in word_frequencies.most_common(500)}
    word_frequencies_sum = sum(word_frequencies.values())
    nouns = [word[0] for word in nltk.pos_tag(word_frequencies.keys()) if word[1] == "NN" or word[1] == "NNS"]
    adjs =  [word[0] for word in nltk.pos_tag(word_frequencies.keys()) if word[1] == "JJ"]

    adj_noun_pmi = pd.DataFrame(list(itertools.product(nouns, adjs))).rename(columns = {0: "noun", 1: "adj"})
    adj_noun_pmi["noun_prob"] = [word_frequencies[word]/word_frequencies_sum for word in adj_noun_pmi["noun"].values]
    adj_noun_pmi["adj_prob"] = [word_frequencies[word]/word_frequencies_sum for word in adj_noun_pmi["adj"].values]

    for i, row in tqdm(adj_noun_pmi.iterrows(), total = len(adj_noun_pmi)):
        adj_noun_pmi.at[i, "adj_noun_prob"] = len([review for review in reviews if row["adj"] in review and row["noun"] in review])/len(reviews)

    adj_noun_pmi["adj_noun_pmi"] = adj_noun_pmi["adj_noun_prob"]/(adj_noun_pmi["noun_prob"]*adj_noun_pmi["adj_prob"])
    adj_noun_pmi["adj_noun_pmi"] = general.normalize_series(adj_noun_pmi["adj_noun_pmi"])
    return adj_noun_pmi[["adj", "noun", "adj_noun_pmi"]]

def get_reviews_by_category(app, categories):
    reviews = general.preprocess_reviews(app=app)
    model, dictionary, index, corpus = general.create_tfidf_model(reviews)
    category_reviews = {}
    for category in categories:
        category_reviews[category] = general.get_tfidf_closest_n([word for word in category.split(" ")], 2000, dictionary, index, model, reviews)
    return category_reviews


def get_noun_category_pmi(categories_reviews):
    noun_category_doc = []
    for category, reviews in categories_reviews.items():
        for i, review in enumerate(reviews):
            for noun in review:
                noun_category_doc.append({"doc":i, "category": category, "noun": noun})
    noun_category_doc = pd.DataFrame(noun_category_doc)

    min_freq = 10
    noun_category_count = noun_category_doc.groupby(['category', 'noun']).count().reset_index()[['category', 'noun', 'doc']].rename(columns = {"doc": "count"})
    noun_category_count = noun_category_count[noun_category_count["count"] > min_freq]

    category_count = pd.DataFrame(noun_category_count.groupby('category')['count'].sum())
    category_count = category_count[category_count["count"] > min_freq]

    noun_count = pd.DataFrame(noun_category_count.groupby('noun')['count'].sum())
    noun_count = noun_count[noun_count["count"] > min_freq]

    noun_count_sum = noun_category_count["count"].sum()

    noun_category_count["probability"] = noun_category_count["count"] / noun_count_sum
    category_count["probability"] = category_count["count"]/category_count["count"].sum()
    noun_count["probability"] = noun_count["count"]/noun_count["count"].sum()

    noun_category_pmi = []
    for i, row in noun_category_count.iterrows():
        noun_category_pmi.append({'category': row['category'],
                    'noun': row['noun'],
                    'noun_cat_pmi': np.log(
                        row.probability / (category_count.loc[row['category']].probability * noun_count.loc[row['noun']].probability))})
    noun_category_pmi = pd.DataFrame(noun_category_pmi)
    noun_category_pmi["noun_cat_pmi"] = general.normalize_series(noun_category_pmi["noun_cat_pmi"])
    return noun_category_pmi