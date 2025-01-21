import argparse
import json
import logging
import random
import time
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

import cvc5

from intrinsic import minimization_ratio


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def build_smt2_formula_from_smt2_constraints(constraints, placeholder):
    """Builds SMT2-Lib formula from constraints' subset, retrieved from GPT-x's response.

    Arguments:
        constraints (list): Constraint subset in string format.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
    
    Returns:
        smt2_formula (str): SMT2-Lib input format string, where assertions corresponding
            to ``constraints`` are inserted in ``placeholder``.
    """
    smt2_assertions = [f"(assert {constraint})" for constraint in constraints]
    assertions = "\n".join(smt2_assertions)
    smt2_formula = placeholder.replace("<ASSERT>", assertions, 1)
    return smt2_formula


def parse_input_formula(solver, formula, formula_name):
    """Parse input formula and load into solver memory.

    Arguments:
        solver (cvc5.Solver): SMT solver.
        formula (str): Input formula in string format.
        formula_name (str): Key for input formula.
    """
    parser = cvc5.InputParser(solver)
    parser.setStringInput(
        cvc5.InputLanguage.SMT_LIB_2_6, formula, formula_name,
    )
    symbol_manager = parser.getSymbolManager()

    while True:
        cmd = parser.nextCommand()
        if cmd.isNull():
            break
        # Invoke the command on the solver and the symbol manager
        cmd.invoke(solver, symbol_manager)

def init_solver(compute_mus=True):
    solver = cvc5.Solver()
    solver.setOption("print-success", "true")
    solver.setOption("produce-models", "true")
    solver.setOption("produce-unsat-cores", "true")
    solver.setOption("unsat-cores-mode", "assumptions")
    solver.setOption("minimal-unsat-cores", str(compute_mus).lower())
    solver.setLogic("QF_SLIA")
    return solver


def check_subset_unsat(solver, subset, placeholder):
    formula = build_smt2_formula_from_smt2_constraints(subset, placeholder)
    solver.resetAssertions()
    parse_input_formula(solver, formula, "smt_formula")
    result = solver.checkSat()
    if result.isUnsat():
        return True
    return False


def stratify_by_interval(all_number_of_constraints):
    """Stratify results based on the number of constraints in the input formula.
    Intervals include 0-10, 10-20, ..., 40-50.

    Arguments:
        all_number_of_constraints (list): List of dictionaries, each recording the
            number of constraints in the input formula, SUS, and MUS.
    
    Returns:
        results_by_intervals (dict): Stratified results.
    """
    are_correct = [True if item['total'] > item['sus'] else False \
                   for item in all_number_of_constraints]

    ratios = [
        minimization_ratio(item['total'], item['sus'], item['mus']) 
        for item in all_number_of_constraints
    ]
    # ratios = [0 for _ in are_correct]

    results_by_intervals = {}
    for is_correct, ratio, number_of_constraints in zip(are_correct, ratios, all_number_of_constraints):
        number_input_constraints = number_of_constraints['total']
        lower_lim = number_input_constraints - (number_input_constraints % 10)
        upper_lim = lower_lim + 10
        key = f'{lower_lim}-{upper_lim}'
        if key in results_by_intervals:
            [stratified_correct, stratified_total], _ratio = results_by_intervals[key]
            if is_correct:
                stratified_correct += 1
            stratified_total += 1
            results_by_intervals[key] = [[stratified_correct, stratified_total], ratio + _ratio]
        else:
            if is_correct:
                results_by_intervals[key] = [[1, 1], ratio]
            else:
                results_by_intervals[key] = [[0, 1], ratio]

        if 'Total' in results_by_intervals:
            [total_correct, total], total_ratio = results_by_intervals['Total']
            if is_correct: total_correct += 1
            total += 1
            results_by_intervals['Total'] = [[total_correct, total], total_ratio + ratio]
        else:
            if is_correct:
                results_by_intervals['Total'] = [[1, 1], ratio]
            else:
                results_by_intervals['Total'] = [[0, 1], ratio]

    results_by_intervals = dict(sorted(results_by_intervals.items()))

    return results_by_intervals



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IntelliSMT with OpenAI GPT-X")

    ## Pipeline arguments
    parser.add_argument("--path_to_data", type=str, default='../dataset',
                        help="Path to processed string constraints dataset file.")
    parser.add_argument("--exrange", type=int, nargs=2, default=[0, 5],
                        help="Range of examples to process: upper-limit not considered.")
    parser.add_argument("--benchmark", type=str, default="Leetcode",
                        choices=["Leetcode", "woorpje", "nornbenchmarks", "kaluza"], help="Benchmark dataset")
    parser.add_argument('--seed', type=int, default=42, help="Set common system-level random seed.")
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'],
                        help=("Evaluation split."))
    parser.add_argument('--max_depth', type=int, default=5, help="Max depth for DFS.")

    ## Unsatisfiability Validator arguments
    parser.add_argument('--timeout', type=int, default=30000,
                        help='Timeout for SMT-2 validator (in milliseconds).')

    args = parser.parse_args()

    # Set random seed.
    random.seed(args.seed)

    # Print arguments
    logger.info(f'Run arguments are: {args}')

    # Load data
    path_to_benchmark = Path(args.path_to_data) / f"unsat.{args.benchmark}.{args.split}.json"
    logger.info(f'Loading data from {path_to_benchmark}')
    with open(path_to_benchmark, 'r') as f:
        data_instances = json.load(f)

    data_instances = data_instances[args.exrange[0]: args.exrange[1]]

    solver = init_solver(False)

    dfs_results, bfs_results = [], []
    for instance in tqdm(data_instances, total=len(data_instances)):
        all_constraints = instance['smt2_constraints']
        placeholder = instance['smt2_formula_placeholder']
        instance_result = check_subset_unsat(solver, all_constraints, placeholder)
        mus = solver.getUnsatCore()

        # DFS-Elimination (Top-Down)
        dfs, dfs_result = deepcopy(all_constraints), False
        for _ in range(args.max_depth):
            index_to_remove = random.randint(0, len(dfs))
            dfs = dfs[:index_to_remove] + dfs[index_to_remove+1:]
            if check_subset_unsat(solver, dfs, placeholder):
                dfs_result = True
                dfs_results.append({
                    'total': deepcopy(len(all_constraints)),
                    'sus': deepcopy(len(dfs)),
                    'mus': deepcopy(len(mus)),
                })
                break

        if not dfs_result:
            dfs_results.append({
                'total': deepcopy(len(all_constraints)),
                'sus': deepcopy(len(all_constraints)),
                'mus': deepcopy(len(mus)),
            })

        # BFS-Elimination (Top-Down)
        bfs_result = False
        for _ in range(args.max_depth):
            index_to_remove = random.randint(0, len(all_constraints))
            bfs = all_constraints[:index_to_remove] + all_constraints[index_to_remove+1:]
            if check_subset_unsat(solver, bfs, placeholder):
                bfs_result = True
                bfs_results.append({
                    'total': deepcopy(len(all_constraints)),
                    'sus': deepcopy(len(bfs)),
                    'mus': deepcopy(len(mus)),
                })
                break

        if not bfs_result:
            bfs_results.append({
                'total': deepcopy(len(all_constraints)),
                'sus': deepcopy(len(all_constraints)),
                'mus': deepcopy(len(mus)),
            })

    for key, _results in zip(
        ['DFS-Elimination', 'BFS-Elimination'], [dfs_results, bfs_results],
    ):
        import statistics
        pct_min = statistics.mean(
            [100*(item['total'] - item['sus'])/item['total'] for item in _results]
        )

        pct_min_corr = statistics.mean(
            [100*(item['total'] - item['sus'])/item['total'] \
             for item in _results \
             if item['total'] != item['sus']]
        )

        print(key)
        print(f"Mean constraint reduction: {pct_min}%")
        print(f"Mean constraint reduction, with correction: {pct_min_corr}%")

        stratified_results = stratify_by_interval(_results)
        print()
        print("Stratification based on intervals of total constraints:")
        print("Interval\tC/N\tAdjusted r\tAbsolute r")
        for key, ratio_counter in stratified_results.items():
            blank_1 = f"{ratio_counter[0][0]}/{ratio_counter[0][1]}"
            try:
                blank_2 = "{:.3f}".format(ratio_counter[1]/ratio_counter[0][0])
            except ZeroDivisionError:
                blank_2 = 0.0
            blank_3 = "{:.3f}".format(ratio_counter[1]/ratio_counter[0][1])
            print(f'{key}\t\t{blank_1}\t{blank_2}\t\t{blank_3}')
