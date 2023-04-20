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

def get_aspect_polarity(aspect_adjs):
    if os.path.exists("data/aspect_polarity.pickle"):
        return pd.read_pickle("data/aspect_polarity.pickle")

    aspect_polarity = pd.Series()
    for aspect in tqdm(aspect_adjs.keys(), total=len(aspect_adjs)):
        aspect_polarity[aspect] = sum([get_sentiwn_score(adj) for adj in aspect_adjs[aspect]])

    norm_aspect_polarity = general.normalize_series(aspect_polarity)
    norm_aspect_polarity.to_pickle("data/aspect_polarity.pickle")
    return norm_aspect_polarity.sort_values(ascending=False)
