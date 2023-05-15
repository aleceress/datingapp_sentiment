import pandas as pd
from tqdm.notebook import tqdm
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import similarities
from rank_bm25 import BM25Okapi
import os 
from datetime import datetime
import spacy
from utils import embeddings, sentiment_extraction, aspect_extraction

def normalize_series(series):
    max = series.max()
    min = series.min()
    normalized_series = series.apply(lambda x: (x-min)/(max-min))
    return normalized_series.sort_values(ascending=False)     

def get_stopwords():
    gist_file = open("data/gist_stopwords.txt", "r")
    try:
        content = gist_file.read()
        stopwords = content.split(",")
    finally:
        gist_file.close()
    return stopwords

def preprocess_reviews(app = "tinder"):
    if app == "tinder":
        reviews_file = pd.read_csv("data/tinder_google_play_reviews.csv").dropna(subset="content")
    elif app == "bumble":
        reviews_file = pd.read_csv("data/DatingAppReviewsDataset.csv").dropna(subset="Review")
        reviews_file = reviews_file[reviews_file.App == "Bumble"].rename(columns={"Review":"content"})
    elif app == "hinge":
        reviews_file = pd.read_csv("data/DatingAppReviewsDataset.csv").dropna(subset="Review")
        reviews_file = reviews_file[reviews_file.App == "Hinge"].rename(columns={"Review":"content"})

    stopwords = get_stopwords()
    reviews = []
    for review in tqdm(reviews_file["content"].values):
        review_words = [] 
        for word in review.split(" "):
            if word not in stopwords and word.isalpha():
                review_words.append(word.lower())
        reviews.append(review_words)
    return reviews

def create_tfidf_model(reviews):
    dictionary = Dictionary(reviews) 
    corpus = [dictionary.doc2bow(line) for line in reviews]
    model = TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(model[corpus], num_features=len(dictionary))
    return model, dictionary, index, corpus

def get_tfidf_closest_n(query, n, dictionary, index, model, reviews):
    query_vec = dictionary.doc2bow(query)
    sims = index[model[query_vec]]
    top_idx = sims.argsort()[-1*n:][::-1]
    return [reviews[i] for i in top_idx]

def bm25_search(reviews, query, bm25):
    query=query.lower()
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    reviews["bm25_score"] = doc_scores
    return reviews[reviews.bm25_score > 4]

def bm25_annotate(queries, app):
    if os.path.isfile(f"data/{app}_annotated.pickle"):
        return pd.read_pickle(f"data/{app}_annotated.pickle")
    
    if app == "tinder":
        reviews = pd.read_csv("data/tinder_google_play_reviews.csv")
        reviews["at"] = reviews["at"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    elif app == "bumble":
        reviews = pd.read_csv("data/DatingAppReviewsDataset.csv")
        reviews = reviews[reviews.App == "Bumble"]
        reviews = reviews.rename(columns = {"Date&Time": "at", "Review":"content", 'Unnamed: 0': "reviewId", "Rating": "score"})
        reviews["at"] = reviews["at"].apply(lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M'))
    elif app == "hinge":
        reviews = pd.read_csv("data/DatingAppReviewsDataset.csv")
        reviews = reviews[reviews.App == "Hinge"]
        reviews = reviews.rename(columns = {"Date&Time": "at", "Review":"content", 'Unnamed: 0': "reviewId", "Rating": "score"})
        reviews["at"] = reviews["at"].apply(lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M'))

    reviews = reviews.dropna(subset=["content"])
    corpus = reviews['content'].tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    annotated_df = pd.DataFrame()
    for query in queries:
        query=query.lower()
        query_reviews = bm25_search(reviews, query, bm25)
        query_reviews["category"] = query.split(" ")[0]
        annotated_df = pd.concat([annotated_df, query_reviews[["reviewId", "content", "score", "at", "bm25_score", "category"]]])

    annotated_df = annotated_df.sort_values(by="bm25_score", ascending=False).drop_duplicates(subset=["reviewId"], keep="first")
    annotated_df.to_pickle(f"data/{app}_annotated.pickle")
    return annotated_df

def convert_to_spacy_doc(nlp, text):
    try: 
        doc = nlp(text)
    except:
        return None
    return doc

