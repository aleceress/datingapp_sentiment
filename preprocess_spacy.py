import spacy
import pandas as pd
from tqdm import tqdm
import gc 
import sys
import os
from utils import general

nlp = spacy.load('en_core_web_lg')

tqdm.pandas()

if sys.argv[1] == "tinder":
    tinder_reviews = pd.read_csv("data/tinder_google_play_reviews.csv")
    offset = int(len(tinder_reviews) / 10)
    
    if not os.path.isdir("data/tinder_spacy"):
        os.mkdir("data/tinder_spacy")
    for i in range(0, 10):
        sample = tinder_reviews.iloc[i * offset : (i+1) * offset]    
        file = f"data/tinder_spacy/spacy_tinder_sample_{i}.pickle"

        print(f"preprocessing {i+1}/10")
        sample["content"] = sample["content"].progress_apply(lambda x: general.convert_to_spacy_doc(nlp, x))
        print(f"saving {i+1}/10")
        sample.to_pickle(file)
        gc.collect()

elif sys.argv[1] == "bumble":
    dating_reviews = pd.read_csv("data/DatingAppReviewsDataset.csv")
    bumble_reviews = dating_reviews[dating_reviews.App == "Bumble"]
    bumble_reviews["Review"] = bumble_reviews["Review"].progress_apply(lambda x: general.convert_to_spacy_doc(nlp, x))
    bumble_reviews.rename(columns = {"Review": "content", "Rating": "score", "Date&Time": "at"}, inplace=True)
    if not os.path.isdir("data/bumble_spacy"):
        os.mkdir("data/bumble_spacy")
    bumble_reviews.to_pickle("data/bumble_spacy/BumbleReviewsSpacy.pickle")

elif sys.argv[1] == "hinge":
    dating_reviews = pd.read_csv("data/DatingAppReviewsDataset.csv")
    hinge_reviews = dating_reviews[dating_reviews.App == "Hinge"]
    hinge_reviews["Review"] = hinge_reviews["Review"].progress_apply(lambda x: general.convert_to_spacy_doc(nlp, x))
    hinge_reviews.rename(columns = {"Review": "content"}, inplace=True)
    if not os.path.isdir("data/bumble_spacy"):
        os.mkdir("data/hinge_spacy")
    hinge_reviews.to_pickle("data/hinge_spacy/HingeReviewsSpacy.pickle")