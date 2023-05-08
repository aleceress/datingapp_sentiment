import pandas as pd
from tqdm.notebook import tqdm
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import similarities

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