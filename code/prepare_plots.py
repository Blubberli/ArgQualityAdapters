import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse


def read_dim2adapter_order(path):
    adapter2order = {}
    data = pd.read_csv(path, sep="\t")
    names = list(data.name.values)
    # replace ibm_rank with quality
    names = [n.replace("ibm_rank", "quality") for n in names]
    # empathie with replace with empathy
    names = [n.replace("empathie", "empathy") for n in names]
    # replace reflexifity with reference
    names = [n.replace("reflexivity", "reference") for n in names]
    adapter2oder = {n: i for i, n in enumerate(names)}
    return adapter2oder


def read_attention(path):
    # return the sum of the attention scores for each adapter over layers and instances
    with open(path, 'rb') as f:
        matrix = pickle.load(f)
    avg_matrix = np.array(matrix).sum(axis=0)
    sum_over_layers = list(avg_matrix.data)
    return sum_over_layers


def create_heatmap(scores, labels, save_path):
    # merge labels and scores into dic
    dim2scores = dict(zip(labels, scores))
    # sort the dic ascending
    sorted_dim2scores = {k: v for k, v in sorted(dim2scores.items(), key=lambda item: item[1], reverse=True)}
    # create a dataframe with the sorted dic
    df = pd.DataFrame.from_dict(sorted_dim2scores, orient='index', columns=["attention"])
    print(df)
    # plot a heatmap (vector) with attention as values and adapter as labels
    sns.set_theme()
    # plot df_mean attention as heatmap but attention should be on the y axis
    # compute xfigsize by dividing the number of adapters by 2 and add 2
    xfigsize = len(labels) / 2 + 2
    # xfigsize = 7
    print("xfigsize: %s" % xfigsize)
    plt.figure(figsize=(xfigsize, 2))
    # ax = sns.heatmap(df, cmap="YlGnBu", cbar=False, yticklabels=True, xticklabels=False)
    ax = sns.heatmap(df.T, annot=True, cmap="YlGnBu", cbar=False, yticklabels=False, xticklabels=True)

    ax.set_ylabel("Attention")
    ax.set_xlabel("Adapter")
    # rotate the labels for 50 degrees and have the xticks exactly in top of where the label starts

    plt.xticks(rotation=50, ha="right")
    # have small ticks on the x axis
    ax.tick_params(axis='x', which='major', labelsize=10, bottom=True)
    # ha

    # tight layout
    plt.tight_layout()
    print("Saving figure to %s" % save_path)
    # save the plot
    plt.savefig(save_path, dpi=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_adapter_path", type=str, required=True,
                        help="Path to the csv file that stores the adapter names and their order in the pretrained model. It is the one that was used also during creating the attention scores.")
    parser.add_argument("--attention_scores_path", type=str, required=True, help="Path to the attention scores which are stored as a pickle file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the heatmap.")
    args = parser.parse_args()
    adapter2oder = read_dim2adapter_order(args.pretrained_adapter_path)
    scores = read_attention(args.attention_scores_path)
    create_heatmap(scores, adapter2oder, args.save_path)
