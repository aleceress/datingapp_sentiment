from gensim.models import KeyedVectors
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

w2v = None

def load_wv2():
    global w2v
    if w2v is None:
        w2v = KeyedVectors.load_word2vec_format("data/word2vec/GoogleNews-vectors-negative300.bin", binary=True)

def get_w2v_embedding(word):
    load_wv2()
    return w2v[word]

def get_aspects_embeddings(aspects):
    if os.path.isfile("data/aspects_embedding.pickle"):
        with open("data/aspects_embedding.pickle", "rb") as f:
            return pickle.load(f)

    load_wv2()

    aspects_embedding = {}
    for aspect in aspects.keys():
        try:
            aspects_embedding[aspect] = w2v[aspect]
        except KeyError:
            continue
    with open("data/aspect_embedding.pickle", "wb+") as f:
            pickle.dump(aspects_embedding, f)
    return aspects_embedding


def get_query_expansion(query, not_query= []):
    load_wv2()
    return [w for w in query] + [i[0].lower().replace("_", " ") for i in w2v.most_similar(positive = [q for q in query], negative = [nq for nq in not_query])]

def get_query_similarities(aspects, query, not_query=[]):
    load_wv2()
    query_embedding = np.zeros(w2v.vector_size)
    query_expansion = get_query_expansion(query, not_query)
    print(f"expansion: {query_expansion}")
    for q in query_expansion:
        for w in q.split(" "):
            try:
                query_embedding += get_w2v_embedding(w)
            except KeyError:
                continue

    query_embedding = query_embedding/len(query_expansion)

    similarity = pd.Series()
    for w, e in aspects.items():
        similarity[w] = cosine_similarity(e.reshape(1,-1), query_embedding.reshape(1,-1))[0][0]
    return similarity