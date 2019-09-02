import argparse
import pickle
import os
from glob import glob
from os.path import basename, isdir, join

import numpy as np
import pandas as pd

from main import main

# root directory containing run directories
# each of the latter should contain predictions in the form:
# large_results_dir/*/results/*.predictions.pickle

large_results_dir = "/home/nik/software/sematext/testlargerouge"

# jsonic dataset and golden summaries
dataset = "/home/nik/datasets/multiling15/multiling_english_lblratio2mod_oversample.json"
goldens = "/home/nik/datasets/multiling15/multiling_english_lblratio2mod_oversample.json.goldens.json"

# should discover this many result files
verify_folds = 5
# execute these types of rouge evaluation
rouge_mode, rouge_ngram, rouge_metric = "Avg", (["rouge-1", "rouge-2"]), "f1"
print_precision = 3

warnings = []

def read_predictions_file(input_path):
    with open(input_path, "rb") as f:
        predictions = pickle.load(f)
    if type(predictions) == list:
        predictions = predictions[0]
    return predictions

def evaluate_prediction_array(predictions, kwargs):
    if type(predictions) is str:
        predictions = read_predictions_file(predictions)
    elif type(predictions) is np.ndarray:
        pass
    else:
        print("Prediction type unhandled:  ", type(predictions))
        exit(1)

    kwargs["input"] = predictions
    if "input_path" in kwargs:
        del kwargs["input_path"]
    return main(**kwargs)


def run_single_folder(**kwargs):
    """Run on a folder containing *.predictions.pickle file(s)
    """
    folder_path = kwargs["input_path"]
    fusion = kwargs["fusion"]
    prediction_pickles = glob(folder_path + "/results/*.predictions.pickle")
    print("Evaluating folder: {} -- got {} prediction files".format(folder_path,
        len(prediction_pickles)))

    if not prediction_pickles:
        print("No predition pickle files found.")
        exit(1)
    if len(prediction_pickles) != verify_folds:
        warnings.append(
            "Expected {} folds, got {} prediction files: {}".format(
                verify_folds, len(prediction_pickles), folder_path))

    run_scores = {x: [] for x in rouge_ngram}
    predictions = [read_predictions_file(path) for path in prediction_pickles]
    if fusion == "early":
        predictions = [np.mean(predictions, axis=0)]

    for p, prediction in enumerate(predictions):
        print("Evaluating prediction {}/{}".format(p+1, len(predictions)))
        kwargs['do_print'] = False
        scores = evaluate_prediction_array(prediction, kwargs)
        scores = scores[rouge_mode]
        for ng in rouge_ngram:
            run_scores[ng].append(scores[ng][rouge_metric])

    # aggregate over prediction files
    if fusion == "late":
        for ng in rouge_ngram:
            run_scores[ng] = np.mean(run_scores[ng])
    return run_scores


def run_large_folder(**kwargs):
    """Run in a directory containing run directories.

    Each run directory should contain a results folder, with *.predictions.pickle prediction ndarray files."""

    all_scores = {}
    large_results_dir = kwargs["input_path"]

    dirs = [d for d in os.listdir(large_results_dir) if isdir(join(large_results_dir, run_id))]
    for r, run_id in enumerate(dirs):
        run_dir = join(large_results_dir, run_id)
        kwargs["input_path"] = run_dir
        print("Evaluating fodler {}/{}: {}".format(r, len(dirs), run_dir))
        run_scores = run_single_folder(**kwargs)
        all_scores[run_id] = run_scores

        # average folds for the run
        # all_scores[run_id] = {
        #     ng: np.mean(run_scores[ng], axis=0)
        #     for ng in rouge_ngram
        # }

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(w)
    return all_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="The evaluation run mode (large / folder / predictions)")
    parser.add_argument("input_path")
    parser.add_argument("dataset_path")
    parser.add_argument("golden_summaries_path")

    parser.add_argument("--fusion", nargs=1, default="late", help="Fusion type for multiple prediction files. If late (default), averages rouge scores. If early, averages predictions.")

    parser.add_argument("--rouge_mode", nargs="*", default=["Avg", "Best"])
    parser.add_argument("--ngram_mode", nargs="*", default=["rouge-1", "rouge-2"])
    parser.add_argument("--metric", nargs="*", default=["f1"])


    args = parser.parse_args()
    #args.fusion = "early"

    if args.mode is None:
        print("No input mode provided, guessing.")
        if not isdir(args.input_path):
            args.mode = "predictions"
        else:
            subdirs = [x for x in os.listdir(args.input_path) if isdir(join(args.input_path, x))]
            if len(subdirs) == 1 and  subdirs[0] == "results":
                args.mode = "folder"
            else:
                args.mode = "large"
        print("Guessed run mode: {} from input path {}".format(args.mode, args.input_path))

    if args.mode == 'large':
        results = run_large_folder(**vars(args))
    elif args.mode == 'folder':
        results = run_single_folder(**vars(args))
    elif args.mode == "predictions":
        kwargs = vars(args)
        kwargs["do_print"] = True
        results = evaluate_prediction_array(args.input_path, kwargs)

    df = pd.DataFrame.from_dict(results, orient='index')
    print(df.round(print_precision).to_string())
    df.to_csv("{}_rouge_scores.csv".format(basename(large_results_dir)))
