"""Contains code for extrinsic evaluation, as in RQ4.
"""
import argparse
import json
import random
import time
from copy import deepcopy
from collections import Counter
from functools import reduce
from pathlib import Path
from tqdm import tqdm

import cvc5

from utils import build_smt2_formula_from_string_constraints


SEED = 42


def compute_mus(formula, placeholder):
    """Compute the minimal unsatisfiable subset (MUS) for a given ``formula``.

    Arguments:
        formula (str): Input formula.
        placeholder (str): SMT2-Lib input placeholder.
    
    Returns:
        (tuple): Tuple of the MUS, and time taken for its computation.
    """
    solver = cvc5.Solver()
    if 'set-logic' not in placeholder:
        solver.setLogic("QF_SLIA")
    solver.setOption("print-success", "true")
    solver.setOption("produce-models", "true")
    solver.setOption("produce-unsat-cores", "true")
    solver.setOption("unsat-cores-mode", "assumptions")
    solver.setOption("produce-difficulty", "true")
    solver.setOption("minimal-unsat-cores", "true")

    parser = cvc5.InputParser(solver)
    parser.setStringInput(
        cvc5.InputLanguage.SMT_LIB_2_6, formula, "smt2_formula"
    )
    symbol_manager = parser.getSymbolManager()

    while True:
        cmd = parser.nextCommand()
        if cmd.isNull():
            break
        # Invoke the command on the solver and the symbol manager
        cmd.invoke(solver, symbol_manager)

    mus_start_time = time.time()
    _ = solver.checkSat()
    unsat_core = solver.getUnsatCore()
    mus_end_time = time.time()

    mus = [str(item) for item in unsat_core]
    mus_time = '{:.3f}'.format((mus_end_time - mus_start_time)*1000)

    return mus, mus_time


def shuffle_and_compute_mus(
        subset, all_constraints, all_smt2_constraints, placeholder, num_shuffles
    ):
    """Compute multiple minimal unsatisfiable subsets (based on ``num_shuffles``),
    each with a different seed to the cvc5 SMT solver.

    Arguments:
        subset (list): Constraint subset from GPT-x's response in string format.
        all_constraints (list): Complete list of constraints in string format.
        all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
            There is one-to-one correspondence with constraints in ``all_constraints``.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
        num_shuffles (int): Number of seeds to use to generate multiple MUSes.
    
    Returns:
        instance (list): List of dictionaries, containing the input constraints,
            its corresponding MUS, and the time taken to compute the MUS.
    """
    instance = {}
    shuffled_instances = None
    for attempt in range(num_shuffles):
        if not shuffled_instances: shuffled_instances = subset
        else:
            random.seed(SEED + attempt)
            shuffled_instances = random.sample(shuffled_instances, len(shuffled_instances))
        formula = build_smt2_formula_from_string_constraints(
            shuffled_instances, all_constraints, all_smt2_constraints, placeholder
        )
        mus, mus_time = compute_mus(formula, placeholder)

        instance[f'Attempt-{attempt}'] = {
            'input_constraints': deepcopy(shuffled_instances),
            'mus': deepcopy(mus),
            'mus_time': deepcopy(mus_time),
        }
    return instance


def compute_muses(
        all_suses, all_constraints, all_smt2_constraints, placeholder
    ):
    """Compute minimal unsatisfiable subsets (MUSes) for both original input formula,
    with different seeds, and all smaller unsatisfiable subsets (SUSes).

    Arguments:
        all_suses (list): Complete list of SUSes in string format.
        all_constraints (list): Complete list of constraints in string format.
        all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
            There is one-to-one correspondence with constraints in ``all_constraints``.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
        num_shuffles (int): Number of seeds to use to generate multiple MUSes.
    
    Returns:
        instance (list): List of dictionaries, containing the input constraints,
            its corresponding MUS, and the time taken to compute the MUS.
    """
    muses_original = shuffle_and_compute_mus(
        all_constraints, all_constraints, all_smt2_constraints, placeholder, num_shuffles=5,
    )
    instance = {'Original': muses_original}

    for idx, sus in enumerate(all_suses):
        instance[f'Candidate-{idx}'] = shuffle_and_compute_mus(
            sus, all_constraints, all_smt2_constraints, placeholder, num_shuffles=5,
        )
    return instance


def build_test_muses(path_to_cache, split):
    """Build MUSes for input formulae and all corresponding SUSes, if not already cached.

    Arguments:
        path_to_cache (str): Path to LLM-generated outputs.
        split (str): Evaluation split.
    """
    path_to_outputs = Path(path_to_cache) / 'smt2_minimizer' / 'sc' / split
    all_outputs = list(Path(path_to_outputs).iterdir())

    with open(str(Path(args.path_to_data) / f'unsat.Leetcode.{args.split}.json'), 'r') as f:
        all_split_instances = json.load(f)

    muses_outputs = []
    for output_file in tqdm(all_outputs):
        with open(output_file, 'r') as f:
            output_json = json.load(f)

        try:
            if output_json['Number of input clauses'] == \
                output_json['SMT2 Minimizer']['Number of constraints to SMT2-Minimizer']:
                continue
        except KeyError:
            if output_json['Number of input constraints'] == \
                output_json['SMT2 Minimizer']['Number of constraints to SMT2-Minimizer']:
                continue

        candidate_keys = [key for key in output_json.keys() if key.startswith('Candidate')]
        all_suses = []
        for key in candidate_keys:
            if output_json[key]['Parsing Status'] == 'Successful':
                if output_json[key]['Validator Response'] == 'UNSAT':
                    all_suses.append(output_json[key]['Parsed Subset'])

        for instance in all_split_instances:
            if instance['path_to_smt2_formula'] == output_json['Path to SMT2 file']:
                break

        muses_instance = compute_muses(
            all_suses, instance['constraints'], instance['smt2_constraints'],
            instance['smt2_formula_placeholder']
        )
        muses_outputs.append(muses_instance)

    with open(str(Path(path_to_cache) / f"{split}_muses.json"), "w") as f:
        json.dump(muses_outputs, f, indent=2)
    
    return muses_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Intrinsic Evaluation")

    parser.add_argument("--path_to_data", type=str, default='../dataset',
                        help='Path to processed string constraints dataset file.')
    parser.add_argument("--path_to_cache", type=str, default='../outputs_all/outputs_claude',
                        help="Path to LLM-generated outputs.")
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'],
                        help=("Evaluation split."))
    parser.add_argument("--sc", action='store_true',
                        help=("Whether to use self-consistency in Explorer LLM."))

    args = parser.parse_args()

    path_to_muses = Path(args.path_to_cache) / f'{args.split}_muses.json'
    try:
        with open(str(path_to_muses), 'r') as f:
            muses_outputs = json.load(f)
    except FileNotFoundError:
        muses_outputs = build_test_muses(args.path_to_cache, args.split)

    if args.sc:
        print_msg = ('Printing evaluation statistics for partially enumerating '
                     'MUSes with IntelliSMT, Self-Consistency')
    else:
        print_msg = ('Printing evaluation statistics for partially enumerating '
                     'MUSes with IntelliSMT, no Self-Consistency')

    sus_counts = dict(Counter([
        len([item for item in instance.keys() if item.startswith('Candidate')]) \
        for instance in muses_outputs
    ]))
    sus_counts = sorted(sus_counts.items(), key=lambda x: x[0])
    for k, v in sus_counts:
        print(f"Number of cases with 1 MUS, but containing {k} SUSes per instance: {v}")

    num_suses_per_instance = {}
    for k in range(1, 6):
        num_suses_per_instance[k] = dict(zip(
            list(range(1, 6)),
            [[0, 0] for _ in range(5)],
        ))

    all_muses_from_original_with_shuffle = []
    all_muses_from_original_without_shuffle = []
    all_muses_from_sus = []
    num_instances_with_unique = 0
    for instance in tqdm(muses_outputs):
        original_mus_without_shuffle = '\n'.join(sorted(instance['Original']['Attempt-0']['mus']))
        all_muses_from_original_without_shuffle.append([original_mus_without_shuffle])

        original_muses = [sorted(item['mus']) for item in list(instance['Original'].values())]
        original_muses_strings = ['\n'.join(item) for item in original_muses]
        original_unique_muses = list(set(original_muses_strings))
        all_muses_from_original_with_shuffle.append(original_unique_muses)

        candidate_keys = [key for key in instance.keys() if key.startswith('Candidate')]
        num_suses = len(candidate_keys)

        candidate_muses = ['\n'.join(sorted(instance[key]['Attempt-0']['mus'])) for key in candidate_keys]
        unique_muses_from_sus = list(set(candidate_muses))
        all_muses_from_sus.append(unique_muses_from_sus)

        if len(unique_muses_from_sus) > 1:
            num_instances_with_unique += 1

        for x in range(1, 6):
            if len(original_unique_muses) == x:
                for y in range(1, 6):
                    if num_suses == y:
                        num_suses_per_instance[x][y][0] += 1
                    if len(candidate_muses) == y:
                        num_suses_per_instance[x][y][1] += len(unique_muses_from_sus)

    for k, v in num_suses_per_instance.items():
        for kk, vv in v.items():
            print(f"Number of instances containing {k} MUSes, and {kk} SUSes: {vv[0]}. "
                  f"These result in {vv[1]} unique MUSes.")
        print()
    
    for idx, original in enumerate([all_muses_from_original_without_shuffle, all_muses_from_original_with_shuffle]):
        if idx == 0:
            print("Without shuffling:")
        else:
            print('With shuffling:')

        common, left_difference, right_difference = 0 , 0 , 0
        for list1, list2 in zip(original, all_muses_from_sus):
            common += len(set(list1).intersection(list2))
            left_difference += len(set(list1).difference(list2))
            right_difference += len(set(list2).difference(list1))

        print(f"  Number of common MUSes between original and SUS: {common}")
        print(f"  Number of MUSes extracted from original, but not from SUS: {left_difference}")
        print(f"  Number of MUSes extracted from SUSes, but not from original: {right_difference}")
        print()

        print(f"Number of instances with multiple non-unique MUSes: {num_instances_with_unique}")
