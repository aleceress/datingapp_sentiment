from nltk.corpus import sentiwordnet as swn
import pandas as pd
from tqdm.notebook import tqdm
from utils import general
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline


def get_sentiwn_score(word):
    synsets = list(swn.senti_synsets(word))
    score = 0.0
    if len(synsets) > 0:
        polarity = pd.DataFrame([{'pos': sw.pos_score(), 
                                  'neg': sw.neg_score(), 
                                  'obj': sw.obj_score()} for sw in synsets])
        avg_polarity = polarity.mean()
        score = avg_polarity['pos'] - avg_polarity['neg']
    return score

def get_aspects_polarity(aspects_adjs, app):
    if os.path.exists(f"data/aspects_polarity_{app}.pickle"):
        return pd.read_pickle(f"data/aspects_polarity_{app}.pickle")

    aspects_polarity = pd.Series()
    for aspect in tqdm(aspects_adjs.keys(), total=len(aspects_adjs)):
        aspects_polarity[aspect] = sum([get_sentiwn_score(adj) for adj in set(aspects_adjs[aspect])])

    norm_aspects_polarity = general.normalize_series(aspects_polarity)
    norm_aspects_polarity.to_pickle(f"data/aspects_polarity_{app}.pickle")
    return norm_aspects_polarity.sort_values(ascending=False)

def get_aspects_polarity_percentage(aspects, aspect_adjs, verbose = True):
    count = 0
    count_positive = 0
    count_negative = 0

    for aspect in aspects:
        for adj in aspect_adjs[aspect]:
            score = get_sentiwn_score(adj)
            if abs(score) > 0.20:
                if verbose: 
                    print(adj, score)
                count = count+1
                if score < 0:
                    count_negative = count_negative+1
                else:
                    count_positive = count_positive+1

    return count_positive/count*100, count_negative/count*100

def get_wordcloud(query_aspects, aspects_adjs):
    query_adjs = []
    for aspect in query_aspects:
        for adj in aspects_adjs[aspect]:
            if abs(get_sentiwn_score(adj)) > 0.4:
                query_adjs.append(adj) 
    
    text = " ".join(query_adjs)

    wordcloud = WordCloud(width=1600, height=800, collocations=False).generate(text)
    wordcloud.recolor(colormap="pink")
    plt.figure(figsize=(10, 5), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show();

def get_reviews_polarities(reviews):
    pos_polarity = []
    neg_polarity = []

    polarity_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", top_k=None, truncation=True)
    for sentence in tqdm(reviews["sentence"].values):
        polarities = polarity_pipeline(sentence)
        for polarity in polarities[0]:
            if polarity["label"] == "NEGATIVE":
                neg_polarity.append(polarity["score"])
            elif polarity["label"] == "POSITIVE":
                pos_polarity.append(polarity["score"])
    reviews["pos"] = pos_polarity
    reviews["neg"] = neg_polarity
    return reviews