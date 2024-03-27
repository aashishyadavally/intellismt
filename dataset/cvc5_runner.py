"""Script for processing string benchmarks with cvc5 SMT solver.

String Benchmarks Reference: https://z3str4.github.io/
The statistics correspond to z3str3 solver in the benchmark analysis tables.

===========================================================================================
  S.No. |   Benchmark                   |   Total   | SAT   |   UNSAT   |   UNK |   T/O   |
-------------------------------------------------------------------------------------------
    1   | PyEX                          |   25421   | 1151  | 13750     | 2     |   10518 |
    2   | SMTLIB25                      |   5913    | 1774  | 3529      | 99    |   511   |
    3   | Pisa                          |   12      | 7     | 4         | 0     |   1     |
    4   | Norn                          |   1027    | 510   | 143       | 0     |   374   |
    5   | Trau Light                    |   100     | 4     | 93        | 0     |   3     |
    6   | LeetCode                      |   2661    | 590   | 1811      | 126   |   134   |
    7   | IBM AppScan                   |   8       | 5     | 0         | 0     |   3     |
    8   | Sloth                         |   24      | 9     | 10        | 0     |   5     |
    9   | Woorpje                       |   809     | 522   | 171       | 14    |   102   |
    10  | Kaluza                        |   47284   | 34303 | 11799     | 215   |   967   |
    11  | stringfuzz                    |   1065    | 606   | 186       | 3     |   270   |
    12  | z3str3 Regression             |   242     | 194   | 44        | 1     |   3     |
    13  | Cashew                        |   394     | 382   | 12        | 0     |   0     |
    14  | JOACO                         |   94      | 17    | 20        | 57    |   0     |
    15  | Stranger                      |   4       | 4     | 0         | 0     |   0     |
    16  | Kausler                       |   120     | 99    | 0         | 0     |   21    |
    17  | BanditFuzz                    |   357     | 20    | 287       | 0     |   50    |
    18  | automatark25                  |   19979   | 11836 | 4411      | 431   |   3301  |
    19  | stringfuzzregexgenerated      |   4170    | 3284  | 32        | 0     |   854   |
    20  | stringfuzzregextransformed    |   10682   | 4346  | 4896      | 1     |   1439  |
===========================================================================================

"""
import argparse
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

import cvc5

import z3

from utils import build_smt2_formula_from_string_clauses, run_fast_scandir


BENCHMARK_EXTS = {
    "Leetcode": [".smt2"],
    "woorpje": [".smt"],
    "stringfuzz": [".smt20", ".smt25"],
    "nornbenchmarks": [".smt2"],
}


def run_cvc5_solver(path_to_smt2_file):
    """Run CVC5 solver on SMT2LIB benchmark.

    Arguments:
        path_to_smt2_file (pathlib.Path): Path to SMTLIB file.
    
    Returns:
        (dict): Processed data instance.
    """
    with open(path_to_smt2_file, 'r') as f:
        smt2_formula = f.read()

    smt2_formula_placeholder = []
    assert_inserted = False
    with open(path_to_smt2_file, 'r') as f:
        for line in f.readlines():
            if not line.startswith('(assert'):
                smt2_formula_placeholder.append(line)
            else:
                if not assert_inserted:
                    smt2_formula_placeholder.append('<ASSERT>')
                    assert_inserted = True

    smt2_formula_placeholder = '\n'.join(smt2_formula_placeholder)

    goal = z3.Goal()
    z3_formula = z3.parse_smt2_string(smt2_formula)
    goal.add(z3_formula)
    algebraic_simplified_formula = z3.Tactic('simplify')(goal)
    assert algebraic_simplified_formula.__len__() == 1
    all_constraints = [algebraic_simplified_formula[0][i].__repr__() \
                       for i in range(algebraic_simplified_formula[0].__len__())]
    all_constraints = [" ".join([_item.strip() for _item in item.split('\n')]) \
                       for item in all_constraints]
    all_smt2_constraints = [algebraic_simplified_formula[0][i].sexpr() \
                            for i in range(algebraic_simplified_formula[0].__len__())]
    all_smt2_constraints = [" ".join([_item.strip() for _item in item.split('\n')]) \
                            for item in all_smt2_constraints]

    if not [item for item in all_constraints if item != "False"]:
        return None

    simplified_smt2_formula = build_smt2_formula_from_string_clauses(
        all_constraints, all_smt2_constraints, smt2_formula_placeholder
    )

    instance = {
        'path_to_smt2_formula': str(path_to_smt2_file),
        'smt2_formula_placeholder': smt2_formula_placeholder,
        'constraints': all_constraints,
        'smt2_constraints': all_smt2_constraints
    }

    for mus_option in ["false", "true"]:
        solver = cvc5.Solver()
        if 'set-logic' not in smt2_formula_placeholder:
            solver.setLogic("QF_SLIA")
        solver.setOption("print-success", "true")
        solver.setOption("produce-models", "true")
        solver.setOption("produce-unsat-cores", "true")
        solver.setOption("unsat-cores-mode", "assumptions")
        solver.setOption("produce-difficulty", "true")
        solver.setOption("minimal-unsat-cores", mus_option)

        parser = cvc5.InputParser(solver)
        parser.setStringInput(
            cvc5.InputLanguage.SMT_LIB_2_6, simplified_smt2_formula, "smt2_formula"
        )
        symbol_manager = parser.getSymbolManager()

        while True:
            cmd = parser.nextCommand()
            if cmd.isNull():
                break
            # Invoke the command on the solver and the symbol manager
            cmd.invoke(solver, symbol_manager)

        unsat_check_start_time = time.time()
        result = solver.checkSat()
        unsat_check_end_time = time.time()

        difficulty = solver.getDifficulty()
        instance['difficulty'] = dict(zip(
            [str(item) for item in difficulty.keys()],
            [str(item) for item in difficulty.values()],
        ))
        if not result.isUnsat(): 
            return None

        unsat_core = solver.getUnsatCore()
        unsat_core_end_time = time.time()

        if mus_option == "false":
            cvc5_assertions = [str(assertion) for assertion in solver.getAssertions()]
            assert len(cvc5_assertions) == len(all_smt2_constraints) == len(all_constraints)
            statistics = solver.getStatistics().get()
            instance = {
                **instance,
                **{
                    'cvc5_assertions': cvc5_assertions, 
                    'unsat_check_time (in ms)': '{:.3f}'.format((unsat_check_end_time - unsat_check_start_time)*1000),
                    'unsat_core': [str(item) for item in unsat_core],
                    'unsat_core_time (in ms)': "{:.3f}".format((unsat_core_end_time - unsat_check_start_time)*1000),
                    'unsat_core_statistics': statistics,
                }
            }
        elif mus_option == "true":
            instance = {
                **instance,
                **{
                    'minimal_unsat_core': [str(item) for item in unsat_core],
                    'mimimal_unsat_core_time (in ms)': "{:.3f}".format((unsat_core_end_time - unsat_check_start_time)*1000),
                }
            }
        solver.resetAssertions()

    return instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create dataset from SMT2 files in the benchmark folder.'
    )
    parser.add_argument('--benchmark_dir', type=str, default="models/Strings",
                        help='Benchmark folder')
    parser.add_argument('--type', type=str, default='Leetcode',
                        choices=['Leetcode', 'stringfuzz', 'woorpje', 'nornbenchmarks'],
                        help='Choice of benchmark.')
    ## SMT2 Arguments
    parser.add_argument('--timeout', default=60000,
                        help='Maximum timeout (in milliseconds).')
    args = parser.parse_args()

    path_to_models = str(Path(args.benchmark_dir) / args.type)

    _, path_to_formulae = run_fast_scandir(path_to_models, BENCHMARK_EXTS[args.type])
    print(f'Number of examples in {args.type} benchmark: {len(path_to_formulae)}')

    path_to_formulae = sorted(path_to_formulae)
    data_instances = []
    for fid, path in tqdm(enumerate(path_to_formulae), total=len(path_to_formulae)):
        try:
            output = run_cvc5_solver(path)
            if output: data_instances.append(output)
        except: continue

    print(f"Number of data instances: {len(data_instances)}")
    print(f"Writing output to unsat.{args.type}.json")
    with open(f'unsat.{args.type}.json', 'w') as f:
        json.dump(data_instances, f, indent=2)
