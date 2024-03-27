"""Combines all ``experiments`` scripts to display Table 1 (RQ1), Table 2 (RQ2),
and overlapping analysis (RQ3).
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from intrinsic_naive import stratify_by_interval as stratify_by_interval_naive
from intrinsic import (
    extract_number_of_constraints,
    stratify_by_interval as stratify_by_interval_gpt
)
from extrinsic import build_test_muses


def compute_intrinsic_naive(path_to_data, split):
    """Compute metrics for intrinsic evaluation in naive baseline.

    Arguments:
        path_to_data (str): Path to processed string constraints dataset file.
        split (str): Evaluation split.
    """
    results_naive = stratify_by_interval_naive(path_to_data, split)
    return results_naive


def compute_intrinsic_gpt(path_to_cache, split, sc):
    """Compute metrics for intrinsic evaluation with GPT-x model.

    Arguments:
        path_to_cache (str): Path to LLM-generated outputs.
        split (str): Evaluation split.
        sc (bool): If True, use self-exploration for decoding. Else, greedy decoding.

    Returns:
        results_by_intervals (dict): Intrinsic evaluation results from GPT-x models.        
    """
    path_to_cache = Path(path_to_cache) / 'smt2_minimizer' / 'sc' / split
    all_outputs = list(Path(path_to_cache).iterdir())
    number_of_constraints = extract_number_of_constraints(all_outputs, sc)
    results_by_intervals = stratify_by_interval_gpt(number_of_constraints)
    return results_by_intervals


def compute_extrinsic_mus_counts(muses_outputs):
    """Compute counts for overalapping analysis in RQ3.

    Arguments:
        muses_outputs (list): List of JSON objects corresponding to GPT-x's outputs.
    
    Returns:
        results_extrinsic (dict): Results for MUSes from original formula, and SUSes.
    """
    all_muses_from_original_with_shuffle = []
    all_muses_from_original_without_shuffle = []
    all_muses_from_sus = []

    for instance in tqdm(muses_outputs):
        original_mus_without_shuffle = '\n'.join(sorted(instance['Original']['Attempt-0']['mus']))
        all_muses_from_original_without_shuffle.append([original_mus_without_shuffle])

        original_muses = [sorted(item['mus']) for item in list(instance['Original'].values())]
        original_muses_strings = ['\n'.join(item) for item in original_muses]
        original_unique_muses = list(set(original_muses_strings))
        all_muses_from_original_with_shuffle.append(original_unique_muses)

        candidate_keys = [key for key in instance.keys() if key.startswith('Candidate')]

        candidate_muses = ['\n'.join(sorted(instance[key]['Attempt-0']['mus'])) \
                           for key in candidate_keys]
        unique_muses_from_sus = list(set(candidate_muses))
        all_muses_from_sus.append(unique_muses_from_sus)

    results_extrinsic = []
    for original in [
        all_muses_from_original_without_shuffle, all_muses_from_original_with_shuffle
        ]:
        common, left_difference, right_difference = 0 , 0 , 0
        for list1, list2 in zip(original, all_muses_from_sus):
            common += len(set(list1).intersection(list2))
            left_difference += len(set(list1).difference(list2))
            right_difference += len(set(list2).difference(list1))

        results_extrinsic.append({
            "Number of common MUSes between original and SUS": common,
            "Number of MUSes extracted from original, but not from SUS": left_difference,
            "Number of MUSes extracted from SUSes, but not from original": right_difference,
        })
    results_extrinsic[0]['key'] = "Without shuffling"
    results_extrinsic[1]['key'] = "With shuffling"

    return results_extrinsic


def print_table_II(results_intrinsic):
    """Print Table II, as in RQ1 (intrinsic evaluation).

    Arguments:
        results_intrinsic (dict): Results from RQ1.
    """
    print("<<< Printing Table II >>>")
    header = f"Approach\t" + "\t".join(results_intrinsic[-1].keys())
    print(header)

    approaches = [
        "Naive", "GPT-3.5 w/o SE", "GPT-3.5 w/ SE", "GPT-4 w/o SE", "GPT-4 w/ SE",
    ]
    for idx, results in enumerate(results_intrinsic):
        if idx == 0:
            results_string = f"{approaches[idx]}\t\t" + \
                "\t".join([f"{v[0]}/{v[1]}" for v in results.values()])
        else:
            results_string = f"{approaches[idx]}\t" + \
                "\t".join([f"{v[0][0]}/{v[0][1]}" for v in results.values()])
        print(results_string)
    print()


def print_table_III(results_intrinsic):
    """Print Table III, as in RQ2 (extrinsic evaluation).

    Arguments:
        results_intrinsic (dict): Results from RQ1, which also contains minimization ratio.
    """
    print("<<< Printing Table III >>>")
    header = f"Approach\t\t" + "\t\t".join(results_intrinsic[-1].keys())
    print(header)

    approaches = ["GPT-3.5 w/ SE", "GPT-4 w/ SE"]
    for idx, results in enumerate([results_intrinsic[2], results_intrinsic[4]]):
        results_string = f"{approaches[idx]}\t | "
        for v in results.values():
            try:
                correct_ratio = "{:.3f}".format(v[1]/v[0][0])
            except ZeroDivisionError:
                correct_ratio = 0.0
            total_ratio = "{:.3f}".format(v[1]/v[0][1]) 
            results_string += f"{correct_ratio}\t{total_ratio} | "
        print(results_string)
    print()


def do_overlapping_analysis(split):
    """Overlapping analysis, as in RQ3 (extrinsic evaluation).

    Arguments:
        split (str): Evaluation split.
    """
    path_to_cache = "../outputs_gpt4"
    path_to_muses = Path(path_to_cache) / f'{split}_muses.json'
    try:
        with open(str(path_to_muses), 'r') as f:
            muses_outputs = json.load(f)
    except FileNotFoundError:
        muses_outputs = build_test_muses(path_to_cache, split)

    results_extrinsic = compute_extrinsic_mus_counts(muses_outputs)
    
    for _results in results_extrinsic:
        print(_results['key'])
        print('-' * len(_results['key']))
        for k, v in _results.items():
            if k != 'key': print(f"  {k}: {v}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Intrinsic Evaluation")

    parser.add_argument("--path_to_data", type=str, default='../dataset',
                        help='Path to processed string constraints dataset file.')
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'],
                        help=("Evaluation split."))

    args = parser.parse_args()
    
    results_intrinsic = [compute_intrinsic_naive(args.path_to_data, args.split)]
    for path_to_cache in ["../outputs_gpt35", "../outputs_gpt4"]:
        for sc in [False, True]:
            results_intrinsic.append(
                compute_intrinsic_gpt(path_to_cache, args.split, sc)
            )

    # Print Table II (RQ1: Intrinsic Evaluation)
    print_table_II(results_intrinsic)

    # Print Table III (RQ2: Extrinsic Evaluation)
    print_table_III(results_intrinsic)

    # Overlapping Analysis (RQ3: Extrinsic Evaluation)
    do_overlapping_analysis(args.split)
