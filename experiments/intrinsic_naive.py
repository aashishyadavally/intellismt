"""
"""
import argparse
import json
import logging
import random
from pathlib import Path


from tqdm import tqdm

import cvc5

from utils import (
    build_smt2_formula_from_string_constraints,
    set_cvc5_options_for_unsat,
    parse_input_formula
)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class UNSATVerifier:
    """A verifier class which checks for the unsatisfiability of a given set
    of constraints.

    Parameters:
        solver (cvc5.Solver): An SMT2 solver instance.
        statistics (dict): Statistics for solving the given set of constraints.
    """
    def __init__(self):
        """Initialize :class: ``intellismt.modules.verifiers.UNSATVerifier``.
        """
        self.solver = cvc5.Solver()
        set_cvc5_options_for_unsat(self.solver)
        self.statistics = None

    def reset(self):
        """Reset all assertions added to the solver, helps with incremental solving.
        """
        self.solver.resetAssertions()

    def check(self, constraints, all_constraints, all_smt2_constraints, placeholder):
        """Checks whether the given set of ``constraints`` is unsatisfiable.

        Arguments:
            constraints (list): Constraint subset from GPT-x's response in string format.
            all_constraints (list): Complete list of constraints in string format.
            all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
                There is one-to-one correspondence with constraints in ``all_constraints``.
            placeholder (str): SMT2-Lib input file string placeholder, where all assertions
                are represented by "<ASSERT>" keyword.
        
        Returns:
            (bool): ``True``, if unsatisfiable, else ``False``.
        """
        smt2_formula = build_smt2_formula_from_string_constraints(
            constraints, all_constraints, all_smt2_constraints, placeholder
        )
        if 'set-logic' not in smt2_formula:
            try: self.solver.setLogic("QF_SLIA")
            except: pass

        parse_input_formula(self.solver, smt2_formula, "smt_formula")

        result = self.solver.checkSat().isUnsat()
        statistics_dict = self.solver.getStatistics().get()
        setattr(self, "statistics", statistics_dict)

        if result: return True
        else:
            self.reset()
            return False


def stratify_by_interval(path_to_data, split):
    """Stratify results based on the number of constraints in the input formula.
    Intervals include 0-10, 10-20, ..., 40-50.

    Arguments:
        path_to_data (str): Path to input dataset.
        split (str): Evaluation split.
    
    Returns:
        stratified_results (dict): Stratified results.
    """
    # Load data
    path_to_benchmark = Path(path_to_data) / f"unsat.LeetCode.{split}.json"
    logger.info(f'Loading data from {path_to_benchmark}')
    with open(path_to_benchmark, 'r') as f:
        data_instances = json.load(f)

    stratified_results = {}

    for instance in tqdm(data_instances, total=len(data_instances)):
        all_clauses = instance['constraints']
        all_smt2_clauses = instance['smt2_constraints']

        subset = [element for element in all_clauses if random.choice([True, False])]

        unsat_verifier = UNSATVerifier()
        # Check whether the clause subset returned by Explorer LLM
        # is indeed UNSAT.
        is_unsat = unsat_verifier.check(
            subset, all_clauses, all_smt2_clauses, instance['smt2_formula_placeholder']
        )

        lower_lim = len(all_clauses) - (len(all_clauses) % 10)
        upper_lim = lower_lim + 10
        key = f'{lower_lim}-{upper_lim}'
        if key in stratified_results:
            [stratified_correct, stratified_total]  = stratified_results[key]
            if is_unsat:
                stratified_correct += 1
            stratified_total += 1
            stratified_results[key] = [stratified_correct, stratified_total]
        else:
            if is_unsat:
                stratified_results[key] = [1, 1]
            else:
                stratified_results[key] = [0, 1]

        if 'Total' in stratified_results:
            total_correct, total = stratified_results['Total']
            if is_unsat: total_correct += 1
            total += 1
            stratified_results['Total'] = [total_correct, total]
        else:
            if is_unsat:
                stratified_results['Total'] = [1, 1]
            else:
                stratified_results['Total'] = [0, 1]
    stratified_results = dict(sorted(stratified_results.items()))
    return stratified_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IntelliSMT with Naive Baseline")

    ## Pipeline arguments
    parser.add_argument("--path_to_data", type=str, default='../dataset',
                        help="Path to processed string constraints dataset file.")
    parser.add_argument('--seed', type=int, default=42, help="Set common system-level random seed.")
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'],
                        help=("Evaluation split."))

    args = parser.parse_args()

    # Set random seed.
    random.seed(args.seed)

    # Print arguments
    logger.info(f'Run arguments are: {args}')

    stratified_results = stratify_by_interval(args.path_to_data, args.split)

    print(f"Stratification based on intervals of total constraints:")
    print(f"Interval\tC/N")
    for key, ratio_counter in stratified_results.items():
        blank_1 = f"{ratio_counter[0]}/{ratio_counter[1]}"
        print(f'{key}\t\t{blank_1}')
