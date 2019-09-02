import argparse
import glob
import json
import pickle
import sys
from os.path import isdir, join

import numpy as np
import pandas as pd
import rouge


"""
Rouge evaluation script
"""


def prepare_results(metric, p, r, f):
    # return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(
    #   metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)
    return 100.0 * p, 100.0 * r, 100.0 * f


def get_rouge_scores(doc_sents, goldens, maxlength, maxtype):
    print(
        "Evaluating ROUGE on {} sentences, {} golden summaries, with  {} maxlen and {} type."
        .format(len(doc_sents), len(goldens), maxlength, maxtype))
    res = {}
    for aggregator in ['Avg', 'Best', 'Individual']:

        res[aggregator] = {}
        # print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l', 'rouge-w'],
            max_n=4,
            limit_length=True,
            length_limit=maxlength,
            # length_limit=100,
            length_limit_type=maxtype,
            # length_limit_type='words',
            apply_avg=apply_avg,
            apply_best=apply_best,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True)

        all_hypothesis = doc_sents
        all_references = goldens

        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            res[aggregator][metric] = {}
            if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                res[aggregator][metric]["precision"] = []
                res[aggregator][metric]["recall"] = []
                res[aggregator][metric]["f1"] = []

                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        # print('\tHypothesis #{} & Reference #{}: '.format(
                        #hypothesis_id, reference_id))

                        res[aggregator][metric]["precision"].append(
                            results_per_ref['p'])
                        res[aggregator][metric]["recall"].append(
                            results_per_ref['r'])
                        res[aggregator][metric]["f1"].append(
                            results_per_ref['f'])

                        #print('\t' + prepare_results(
                        #    metric, results_per_ref['p'][reference_id],
                        #    results_per_ref['r'][reference_id],
                        #    results_per_ref['f'][reference_id]))
                #print()
            else:
                res[aggregator][metric]["precision"] = results['p']
                res[aggregator][metric]["recall"] = results['r']
                res[aggregator][metric]["f1"] = results['f']
                # print(
                #     prepare_results(metric, results['p'], results['r'],
                #                     results['f']))
        #print()
    return res


def print_results(scores, rouge_mode, ngram_mode, metric):
    print_scores = {}
    for aggr in scores:
        if aggr not in rouge_mode:
            continue
        if aggr not in print_scores:
            print_scores[aggr] = {}

        for rmetr in scores[aggr]:
            if rmetr not in ngram_mode:
                continue
            if rmetr not in print_scores[aggr]:
                print_scores[aggr][rmetr] = {}

            for evmetr in scores[aggr][rmetr]:
                if evmetr not in metric:
                    continue
                print_scores[aggr][rmetr][evmetr] = \
                    scores[aggr][rmetr][evmetr]

    print("Categories:", rouge_mode)
    df = pd.DataFrame.from_dict(scores["Avg"], orient='index')
    print(df.round(3).to_string())


def get_selected_sentences(dataset_path, predictions):
    """Retrieves selected input dataset sentences given extractive prediction scores."""
    if len(predictions.shape) > 1:
        # get selected sentence indexes, if preditions are probabilities or scores
        predictions = np.argmax(predictions, axis=1)

    selected_sents = np.where(predictions == 1)[0]

    # get sentences themselves
    with open(dataset_path) as f:
        dataset = json.load(f)
    selected_sents = [dataset['data']['test'][idx] for idx in selected_sents]
    # group by document index
    doc_sents = {}
    for dat in selected_sents:
        sent = dat['text']
        doc_index = dat['document_index']
        print(doc_index)
        if doc_index not in doc_sents:
            doc_sents[doc_index] = []
        doc_sents[doc_index].append(sent)
    return doc_sents, predictions, selected_sents


def parse_input_json(input_path):
    """Get an input spassummaries json. TODO abstract."""
    with open(input_path) as f:
        data = json.load(f)
    res = {}
    for datum in data["golden"]:
        res[len(res)] = datum["summaries"]
    return res

def main(**kwargs):
    input_arg = kwargs["input"]
    dataset_path = kwargs["dataset_path"]
    golden_summaries_path = kwargs["golden_summaries_path"]
    do_print = kwargs["do_print"]
    rouge_mode = kwargs["rouge_mode"]
    ngram_mode = kwargs["ngram_mode"]
    metric = kwargs["metric"]

    ngram_mode = [ngram_mode] if type(ngram_mode) is not list else ngram_mode
    metric = [metric] if type(metric) is not list else metric

    if type(input_arg) is np.ndarray:
        doc_sents, predictions, sels = get_selected_sentences(dataset_path, input_arg)
    elif type(input_arg) is str and input_arg.endswith(".json"):
        doc_sents = parse_input_json(input_arg)
    else:
        print("Undefined input of type: {}: {}".format(type(input_arg), input_arg))
        exit(1)


    doc_text = {}
    # merge selected sentences
    for doc_idx in doc_sents:
        sents = [s.strip() for s in doc_sents[doc_idx]]
        sents = [s if s.endswith(".") else s + "." for s in sents]
        doc_text[doc_idx] = " ".join(sents)

    with open(golden_summaries_path) as f:
        goldens = json.load(f)

    doc_goldens = {}
    doc_text_goldens = {}
    num_no_assign = 0
    for gold in goldens['golden']:
        summs = gold['summaries']
        doc_index = gold['document_index']
        if doc_index in doc_text_goldens:
            print("Encountered already existing doc index in goldens: ",
                  doc_index)
            exit(1)
        doc_goldens[doc_index] = summs
        doc_text_goldens[doc_index] = " ".join(summs)
        if doc_index not in doc_text:
            doc_text[doc_index] = ""
            num_no_assign += 1
    if num_no_assign > 0:
        print(
            "[!] A total of {} out of {} document indexes  have no assigned summary! -- setting empty string"
            .format(num_no_assign, len(goldens['golden'])))

    doc_keys = list(doc_text.keys())
    doc_text = [doc_text[k] for k in doc_keys]
    doc_text_goldens = [doc_text_goldens[k] for k in doc_keys]

    maxlength, maxtype = 100, "words"
    scores = get_rouge_scores(doc_text, doc_text_goldens, maxlength, maxtype)
    with open("results.pickle", "wb") as f:
        pickle.dump(scores, f)

    if do_print:
        print_results(scores, rouge_mode, ngram_mode, metric)
        # print_results(scores, ["Best"])

    return scores


if __name__ == "__main__":
    """
    Arguments:
    predictions: 
    dataset_path: path to the json dataset serialization. The structure below should be present:
                  {"data": {"test":[{"text":"bla..", "document_index": 0}, .... ]}}
    golden_summaries_path: path to the golden summaries json dataset serialization. The structure below should be present:
                  {"golden": [{"summary": ["summary sent1", ...], "document_index": 0}, ...]}
    """
    parser = argparse.ArgumentParser(description='')
    # classification results file, to pick which sentences where classified as summary parts
    parser.add_argument('input',
                        help="Numpy ndarray with proba scores or prediction indexes" + \
                         " or json file of the form" + \
                         """{"goldens":[{"summaries:"["summary one", "summary two"], "document_index":0}, ...]}""")

    # the dataset path corresponding to the results
    parser.add_argument('dataset_path', help="Path to the dataset json file of the test data.")
    # the golden summaries path
    parser.add_argument('golden_summaries_path', help="Path to the golden summaries fo the test data.")
    parser.add_argument('--do_print',
                        action="store_true",
                        help="Print results or not (default true)",
                        dest="do_print",
                        default=True)
    parser.add_argument('-rouge_mode', help="Avg Best or Individual", default="Avg")
    parser.add_argument('-ngram_mode', help="rouge-1, ..., rouge-4, rouge-l, rouge-w", default=["rouge-1", "rouge-2"])
    parser.add_argument('-metric', help="precision, recall or f1", default=["f1"])
    args = parser.parse_args()
    main(**vars(args))
