import spacy
import pandas as pd
from tqdm import tqdm
import gc 

nlp = spacy.load('en_core_web_sm')
dataset_path = "data/tinder_google_play_reviews.csv"
tinder_reviews = pd.read_csv(dataset_path)
tinder_reviews = tinder_reviews[["reviewId", "content"]]

def convert_to_spacy_doc(nlp, text):
    try: 
        doc = nlp(text)
    except:
        return None
    return doc

tqdm.pandas()

offset = int(len(tinder_reviews) / 10)
for i in range(0, 10):
    sample = tinder_reviews.iloc[i * offset : (i+1) * offset]    
    file = f"data/tinder_spacy/spacy_tinder_sample_{i}.pickle"

    print(f"preprocessing {i+1}/10")
    sample["content"] = sample["content"].progress_apply(lambda x: convert_to_spacy_doc(nlp, x))
    print(f"saving {i+1}/10")
    sample.to_pickle(file)
    gc.collect()

