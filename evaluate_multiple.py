import os
from glob import glob
from os.path import basename, isdir, join

import numpy as np
import pandas as pd

from main import main

# root directory containing run directories
# each of the latter should contain predictions in the form:
# large_results_dir/*/results/*.predictions.pickle
large_results_dir = "/root/multiling/nlp-semantic-augmentation/multiling_runs/ovs_multiling_fasttextpretr_freqsem_5mlp512_variable_sem_trans"
# jsonic dataset and golden summaries
dataset = "/root/multiling/nlp-semantic-augmentation/multiling_english_lblratio2mod_oversample.json"
goldens = "/root/multiling/nlp-semantic-augmentation/multiling_english_lblratio2mod_oversample.json.goldens.json"

# should discover this many result files
verify_folds = 5
# execute these types of rouge evaluation
rouge_mode, rouge_ngram, rouge_metric = "Avg", (["rouge-1", "rouge-2"]), "f1"
print_precision = 3

all_scores = {}
warnings = []
for run_id in os.listdir(large_results_dir):
    if not isdir(join(large_results_dir, run_id)):
        continue

    run_dir = join(large_results_dir, run_id)
    prediction_pickles = glob(run_dir + "/results/*.predictions.pickle")
    print("Got {} prediction files from folder: {}".format(len(prediction_pickles), run_dir))
    if len(prediction_pickles) != verify_folds:
        warnings.append("Expected {} folds, got {} prediction files: {}".format(verify_folds, len(prediction_pickles), run_dir))

    run_scores = {x: [] for x in rouge_ngram}
    for predfile in prediction_pickles:
        scores = main(predfile, dataset, goldens)[rouge_mode]
        for ng in rouge_ngram:
            run_scores[ng].append(scores[ng][rouge_metric])

    print(run_scores)
    # average folds
    all_scores[run_id] = {ng: np.mean(run_scores[ng], axis=0) for ng in rouge_ngram}

df = pd.DataFrame.from_dict(all_scores, orient='index')
print(df.round(print_precision).to_string())
df.to_csv("{}_rouge_scores.csv".format(basename(large_results_dir)))

if warnings:
    for w in warnings:
        print(w)
