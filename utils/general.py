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