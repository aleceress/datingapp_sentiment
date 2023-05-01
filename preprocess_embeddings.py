import pandas as pd
from tqdm import tqdm
import gc 
import sys
import os
from utils import embeddings, general

app = sys.argv[1]

reviews_files = os.listdir(f"data/{app}_spacy/")
if not os.path.isdir(f"data/{app}_embeddings/"):
    os.mkdir(f"data/{app}_embeddings/")

for i, filename in enumerate(reviews_files):
    reviews_embeddings = []
    print(f"processing reviews {i+1}/{len(reviews_files)}")
    reviews = pd.read_pickle(f"data/{app}_spacy/{filename}")
    for _, review in tqdm(reviews.iterrows(), total = len(reviews)):
        if review.content != None:
            for sentence in review.content.sents:
                stopwords = general.get_stopwords()
                sentence_embedding = embeddings.get_text_embedding(" ".join([word for word in str(sentence).split(" ") if word not in stopwords]))
                reviews_embeddings.append((review.reviewId, str(sentence), sentence_embedding, review.score, review["at"]))
    pd.DataFrame(reviews_embeddings).rename(columns= {0: "id", 1:"sentence", 2: "sentence_embedding", 3: "score", 4: "time"}).to_pickle(f"data/{app}_embeddings/{app}_{i}.pickle")
    gc.collect()
