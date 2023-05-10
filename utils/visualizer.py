import numpy as np
import matplotlib.pyplot as plt

def plot_apps_polarity(apps_polarity, aspects, figsize):
    x = np.arange(len(aspects))
    width = 0.2
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=figsize)

    apps_polarity = apps_polarity.set_index("Aspect")
    ax1.bar(x-0.2, apps_polarity.loc[[f"{aspect} pos" for aspect in aspects], ["Tinder"]].values.reshape(-1), width, color='deeppink')
    ax1.bar(x, apps_polarity.loc[[f"{aspect} pos" for aspect in aspects], ["Bumble"]].values.reshape(-1), width, color='gold')
    ax1.bar(x+0.2, apps_polarity.loc[[f"{aspect} pos" for aspect in aspects], ["Hinge"]].values.reshape(-1), width, color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(aspects)
    ax1.set_xlabel("Aspects")
    ax1.set_ylabel("Positive Polarity Score %")
    ax1.legend(["Tinder", "Bumble", "Hinge"])

    ax2.bar(x-0.2, apps_polarity.loc[[f"{aspect} neg" for aspect in aspects], ["Tinder"]].values.reshape(-1), width, color='deeppink')
    ax2.bar(x, apps_polarity.loc[[f"{aspect} neg" for aspect in aspects], ["Bumble"]].values.reshape(-1), width, color='gold')
    ax2.bar(x+0.2, apps_polarity.loc[[f"{aspect} neg" for aspect in aspects], ["Hinge"]].values.reshape(-1), width, color='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(aspects)
    ax2.set_xlabel("Aspects")
    ax2.set_ylabel("Negative Polarity Score % ")
    ax2.legend(["Tinder", "Bumble", "Hinge"])

    ax3.bar(x-0.2, apps_polarity.loc[[f"{aspect} avg" for aspect in aspects], ["Tinder"]].values.reshape(-1), width, color='deeppink')
    ax3.bar(x, apps_polarity.loc[[f"{aspect} avg" for aspect in aspects], ["Bumble"]].values.reshape(-1), width, color='gold')
    ax3.bar(x+0.2, apps_polarity.loc[[f"{aspect} avg" for aspect in aspects], ["Hinge"]].values.reshape(-1), width, color='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(aspects)
    ax3.set_xlabel("Aspects")
    ax3.set_ylabel("Avg Polarity Scores")
    ax3.legend(["Tinder", "Bumble", "Hinge"])

    plt.subplots_adjust(wspace=0.3)

    plt.show()

def plot_time_avg_score_category(reviews, queries, figsize=(8,10)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.xlabel("year")
    plt.ylabel("avg star score")
    for category in queries:
        category_name = category.split(" ")[0]
        plt.plot(reviews[reviews.category == category_name].groupby(reviews["at"].map(lambda x : x.year))["score"].mean(), label = category_name)
    plt.legend()
    plt.show()