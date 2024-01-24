# %%
import json
from typing import List, Tuple, Optional, Dict
import open_ended_prompting as open_ended_prompting
import os
import torch as t
import numpy as np
import matplotlib.pyplot as plt

# %%

def gather_results(n: int, path: str, oneprompt=False, cheat=False):
    all_results = []
    all_flipped_results = []
    all_mixed_results = []
    for file in os.listdir(path):
        if f"n={n}_" not in file:
            continue
        # print(file)
        if ("cheat" in file)!=cheat:
            continue
        if n > 0 and oneprompt and f"oneprompt=True" not in file:
            continue
        if n > 0 and not oneprompt and f"oneprompt=True" in file:
            continue
        with open(os.path.join(path,file), "r") as f:
            file_results = json.load(f)
        # print(len(file_results)/120)
        if 'flipped_' in file:
            all_flipped_results += file_results
        elif 'mixed_' in file:
            all_mixed_results += file_results
        else:
            all_results += file_results
    return all_results, all_flipped_results, all_mixed_results

# %%

def results_over_n(subfolder, oneprompt=False, dataset_size=120, cheat=False, verbose=False):
    x_axis = []
    pos_syc = []
    neg_syc = []
    mix_syc = []
    for i in range(13):
        if verbose:
            print(f"n={i}")
        results, flipped_results, mixed_results = gather_results(i, f"results/{subfolder}/", oneprompt=oneprompt, cheat=cheat)
        if len(results) == 0:
            if verbose:
                print("no results")
            continue
        if i==0:
            flipped_results = results
            mixed_results = results
        if verbose:
            print(f"count: {len(results)/dataset_size}, {len(flipped_results)/dataset_size}")
            print(f"few-shot: {sum(results)/len(results)}")
            print(f"flipped: {sum(flipped_results)/len(flipped_results)}")
            print(f"mixed: {sum(mixed_results)/len(mixed_results)}")
            print()
        x_axis.append(i)
        pos_syc.append(sum(results)/len(results))
        neg_syc.append(sum(flipped_results)/len(flipped_results))
        mix_syc.append(sum(mixed_results)/len(mixed_results))
    return x_axis, pos_syc, neg_syc, mix_syc

# %%
if __name__ == "__main__":
    verbose = True
    x_axis, pos_syc, neg_syc, mix_syc = results_over_n('casing', oneprompt=False, verbose=verbose)
    x_axis_op, pos_syc_op, neg_syc_op, mix_syc_op = results_over_n('casing', oneprompt=True, verbose=verbose)

    plt.plot(x_axis, pos_syc, label="few-shot", color="blue", linestyle="solid")
    plt.plot(x_axis, neg_syc, label="flipped", color="orange", linestyle="solid")
    plt.plot(x_axis, mix_syc, label="random", color="green", linestyle="solid")

    plt.plot(x_axis_op, pos_syc_op, label="few-shot (oneprompt)", color="blue", linestyle="dashed")
    plt.plot(x_axis_op, neg_syc_op, label="flipped (oneprompt)", color="orange", linestyle="dashed")
    plt.plot(x_axis_op, mix_syc_op, label="random (oneprompt)", color="green", linestyle="dashed")

    plt.rcParams.update({'font.size': 16})
    # plt.scatter([0], [0.7313], label="cheating", color="red")
    plt.title("Mistral-7b-Instruct-v0.1 on PU")
    plt.xlabel("number of examples in few-shot prompt")
    plt.ylabel("Uppercase Accuracy")
    # plt.ylim(.48, .52)
    plt.legend()
    plt.show()

# %%

# # plt.plot(pos_syc, neg_syc, '-o', colorscale='viridis')
# cmap = plt.get_cmap('viridis')
# num_points = len(x_axis)

# # Plotting line segments and changing color
# for i in range(num_points - 1):
#     plt.plot(pos_syc[i:i+2], neg_syc[i:i+2], color=cmap(i / (num_points - 1)))

# for x, y, label in zip(pos_syc, neg_syc, x_axis):
#     plt.scatter(x,y, label=f"n={label}", color='black')
#     plt.text(x,y,label)

# # plt.scatter(pos_syc, neg_syc)
# plt.show()

# %%

# QUESTION: do I really need to be testing on 120 examples?
# let's see the stdv of the results within each file (which is within the 120 test problems)
def gather_stdvs(n: int, path: str):
    all_stdvs = []
    for file in os.listdir(path):
        if not file.endswith(".json"):
            continue
        if f"_n={n}_" not in file:
            continue
        with open(os.path.join(path,file), "r") as f:
            file_results = json.load(f)
        # print(len(file_results)/120)
        all_stdvs.append(np.std(np.array(file_results)))
    return all_stdvs

# %%
if __name__ == "__main__":
    print("STANDARD DEVIATIONS WITHIN TEST SET")
    for i in range(13):
        print(f"n={i}")
        stdvs = gather_stdvs(i, "results/7/")
        if len(stdvs) == 0:
            print("no results")
            continue
        print(f"count: {len(stdvs)}")
        print(f"mean few-shot stdv: {np.mean(np.array(stdvs))/np.sqrt(120)}")
        print()

# %%
# What about different prompts? How many times should I be sampling n-shot prompts for each n?

def gather_result_means(n: int, path: str, oneprompt=False):
    all_results = []
    all_flipped_results = []
    all_mixed_results = []
    for file in os.listdir(path):
        if f"_n={n}_" not in file:
            continue
        if n > 0 and oneprompt and f"oneprompt=True" not in file:
            continue
        if n > 0 and not oneprompt and f"oneprompt=True" in file:
            continue
        with open(os.path.join(path,file), "r") as f:
            file_results = json.load(f)
        # print(len(file_results)/120)
        score = sum(file_results)/len(file_results)
        if 'flipped' in file:
            all_flipped_results.append(score)
        elif 'mixed' in file:
            all_mixed_results.append(score)
        else:
            all_results.append(score)
    return all_results, all_flipped_results, all_mixed_results

# %%

if __name__ == "__main__":
    print("STANDARD DEVIATIONS BETWEEN PROMPTS")
    for i in range(13):
        print(f"n={i}")
        results, flipped_results, mixed_results = gather_result_means(i, "results/8/")
        if len(results) == 0:
            print("no results")
            continue
        if i==0:
            flipped_results = results
            mixed_results = results
        print(f"count: {len(results)}")
        print(f"stdv between few-shot prompts: {100*np.std(np.array(results))}") # type: ignore
        print(f"stdv between flipped prompts: {100*np.std(np.array(flipped_results))}") # type: ignore
        print(f"stdv between mixed prompts: {100*np.std(np.array(mixed_results))}") # type: ignore
        print(f"stdv in mean of few-shot prompts: {100*np.std(np.array(results))/np.sqrt(len(results))}") # type: ignore
        print(f"stdv in mean of flipped prompts: {100*np.std(np.array(flipped_results))/np.sqrt(len(flipped_results))}") # type: ignore
        print(f"stdv in mean of mixed prompts: {100*np.std(np.array(mixed_results))/np.sqrt(len(mixed_results))}") # type: ignore
        print()

# %%
