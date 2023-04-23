from nltk.corpus import sentiwordnet as swn
import pandas as pd
from tqdm.notebook import tqdm
from utils import general
import os


def get_sentiwn_score(word):
    synsets = list(swn.senti_synsets(word))
    score = 0.0
    if len(synsets) > 0:
        polarity = pd.DataFrame([{'pos': sw.pos_score(), 
                                  'neg': sw.neg_score(), 
                                  'obj': sw.obj_score()} for sw in synsets])
        avg_polarity = polarity.mean()
        score = abs(avg_polarity['pos'] + avg_polarity['obj']) - (avg_polarity['neg'] + avg_polarity['obj'])
    return score

def get_aspects_polarity(aspects_adjs):
    if os.path.exists("data/aspects_polarity.pickle"):
        return pd.read_pickle("data/aspects_polarity.pickle")

    aspects_polarity = pd.Series()
    for aspect in tqdm(aspects_adjs.keys(), total=len(aspects_adjs)):
        aspects_polarity[aspect] = sum([get_sentiwn_score(adj) for adj in aspects_adjs[aspect]])

    norm_aspects_polarity = general.normalize_series(aspects_polarity)
    norm_aspects_polarity.to_pickle("data/aspects_polarity.pickle")
    return norm_aspects_polarity.sort_values(ascending=False)

def get_aspects_polarity_percentage(aspects, aspect_adjs, verbose = True):
    count = 0
    count_positive = 0
    count_negative = 0

    for aspect in aspects:
        for adj in aspect_adjs[aspect]:
            score = get_sentiwn_score(adj)
            if abs(score) > 0.40:
                if verbose: 
                    print(adj, score)
                count = count+1
                if score < 0:
                    count_negative = count_negative+1
                else:
                    count_positive = count_positive+1

    return count_positive/count*100, count_negative/count*100