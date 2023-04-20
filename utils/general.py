def normalize_series(series):
    max = series.max()
    min = series.min()
    normalized_series = series.apply(lambda x: (x-min)/(max-min))
    return normalized_series.sort_values(ascending=False)     