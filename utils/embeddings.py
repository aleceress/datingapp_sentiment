import gensim.downloader
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from utils import general
from tqdm import tqdm

glove_twitter = None

def load_glove():
    global glove_twitter
    if glove_twitter is None:
        glove_twitter = gensim.downloader.load('glove-twitter-200')

def get_glove_embedding(word):
    load_glove()
    return glove_twitter[word]

def get_aspects_embeddings(aspects, app):
    if os.path.isfile(f"data/aspects_embedding_{app}.pickle"):
        with open(f"data/aspects_embedding_{app}.pickle", "rb") as f:
            return pickle.load(f)

    load_glove()

    aspects_embedding = {}
    for aspect in aspects.keys():
        try:
            aspects_embedding[aspect] = glove_twitter[aspect]
        except KeyError:
            continue
    with open(f"data/aspects_embedding_{app}.pickle", "wb+") as f:
            pickle.dump(aspects_embedding, f)
    return aspects_embedding


def get_query_expansion(query, not_query= []):
    load_glove()
    stopwords = general.get_stopwords()
    return [w for w in query] + [i[0].lower().replace("_", " ") for i in glove_twitter.most_similar(positive = [q for q in query], negative = [nq for nq in not_query], topn=5) if i not in stopwords]

def get_text_embedding(text):
    load_glove()
    text_embedding = np.zeros(glove_twitter.vector_size)
    words = text.split(" ")
    for word in words:
        try:
            text_embedding += get_glove_embedding(word.lower())
        except KeyError:
            continue
    return text_embedding/len(words)

def get_query_similarities(aspects, query, not_query=[], threshold = 0):
    load_glove()
    query_expansion = get_query_expansion(query, not_query)
    query_embedding = np.zeros(glove_twitter.vector_size)

    print(f"expansion: {query_expansion}")
    for q in query_expansion:
        query_embedding += get_text_embedding(q)

    query_embedding = query_embedding/len(query_expansion)

    similarity = pd.Series()
    for w, e in tqdm(aspects.items(), total = len(aspects)):
        cos_similarity = cosine_similarity(e.reshape(1,-1), query_embedding.reshape(1,-1))[0][0]
        if cos_similarity > threshold:
            similarity[w] = cosine_similarity(e.reshape(1,-1), query_embedding.reshape(1,-1))[0][0]
    return similarity