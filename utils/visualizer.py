import numpy as np
import matplotlib.pyplot as plt

def plot_apps_polarity(apps_polarity, start_aspect, end_aspect, figsize):
    x = np.arange((end_aspect-start_aspect)/2)
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.bar(x-0.2, apps_polarity.iloc[list(range(start_aspect,end_aspect,2)), 1].values, width, color='deeppink')
    ax1.bar(x, apps_polarity.iloc[list(range(start_aspect,end_aspect,2)), 2].values, width, color='gold')
    ax1.bar(x+0.2, apps_polarity.iloc[list(range(start_aspect,end_aspect,2)), 3].values, width, color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([" ".join(label.split(" ")[:-1]) for label in apps_polarity.iloc[list(range(start_aspect,end_aspect,2)), 0].values])
    ax1.set_xlabel("Aspects")
    ax1.set_ylabel("Positive Polarity Scores")
    ax1.legend(["Tinder", "Bumble", "Hinge"])

    ax2.bar(x-0.2, apps_polarity.iloc[list(range(start_aspect+1,end_aspect,2)), 1].values, width, color='deeppink')
    ax2.bar(x, apps_polarity.iloc[list(range(start_aspect+1,end_aspect,2)), 2].values, width, color='gold')
    ax2.bar(x+0.2,apps_polarity.iloc[list(range(start_aspect+1,end_aspect,2)), 3].values, width, color='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels([" ".join(label.split(" ")[:-1]) for label in apps_polarity.iloc[list(range(start_aspect,end_aspect,2)), 0].values])
    ax2.set_xlabel("Aspects")
    ax2.set_ylabel("Negative Polarity Scores")
    ax2.legend(["Tinder", "Bumble", "Hinge"])

    plt.subplots_adjust(wspace=0.3)

    plt.show()