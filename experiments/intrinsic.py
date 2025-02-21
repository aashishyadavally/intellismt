"""Contains code for experiments in intrinsic evaluation (RQ1).
"""
import argparse
import json
from pathlib import Path


def minimization_ratio(
        num_all_constraints, num_subset_constraints, num_mus_constraints
    ):
    """Computes minimization ratio. For a constraint system with ``num_all_constraints``
    number of constraints, ``num_subset_constraints`` number of constraints in the SUS,
    and ``num_mus_constraints`` number of constraints in the MUS:

        m = (num_all_constraints - num_subset_constraints) 
            ----------------------------------------------
              (num_all_constraints - num_mus_constraints)
    
    If m -> 0, SUS is closer to the input formula.
    If m -> 1, SUS is closer to the MUS.

    If the input formula, in itself is the MUS, m = 1.

    Arguments:
        num_all_constraints (int): Number of constraints in input formula.
        num_subset_constraints (int): Number of constraints in the SUS.
        num_mus_constraints (int): Number of constraints in the MUS.
    
    Returns:
        (float): Minimization ratio.
    """
    if num_all_constraints == num_mus_constraints:
        return 1.0
    else:
        return (num_all_constraints - num_subset_constraints) / \
            (num_all_constraints - num_mus_constraints)


def minimization_ratio_sus(num_all_constraints, num_subset_constraints):
    """Computes minimization ratio for SUSes.

        m = (num_all_constraints - num_subset_constraints) 
            ----------------------------------------------
                       num_all_constraints

    Arguments:
        num_all_constraints (int): Number of constraints in input formula.
        num_subset_constraints (int): Number of constraints in the SUS.
    
    Returns:
        (float): Minimization ratio.
    """
    if num_all_constraints == 0:
        return 1.0
    else:
        return (num_all_constraints - num_subset_constraints) / \
            (num_all_constraints)


def stratify_by_interval(all_number_of_constraints):
    """Stratify results based on the number of constraints in the input formula.
    Intervals include 0-10, 10-20, ..., 40-50.

    Arguments:
        all_number_of_constraints (list): List of dictionaries, each recording the
            number of constraints in the input formula, SUS, and MUS.
    
    Returns:
        results_by_intervals (dict): Stratified results.
    """
    are_correct = [True if item['to_explorer'] > item['to_minimizer'] else False \
                   for item in all_number_of_constraints]

    ratios = [minimization_ratio_sus(
                item['to_explorer'], item['to_minimizer'],
              ) for item in all_number_of_constraints]

    results_by_intervals = {}
    for is_correct, ratio, number_of_constraints in zip(are_correct, ratios, all_number_of_constraints):
        number_input_constraints = number_of_constraints['to_explorer']
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


def stratify_by_minimization_ratio(all_number_of_constraints):
    """Stratify C/N metrics based on minimization ratio. Intervals include
    0.0-0.1, 0.1-0.2, ..., 0.9-1.0.

    Arguments:
        all_number_of_constraints (list): List of dictionaries, each recording the
            number of constraints in the input formula, SUS, and MUS.

    Returns:
        (tuple): Stratified results, and number of instances exactly reducing to MUSes.
    """
    are_correct = [True if item['to_explorer'] > item['to_minimizer'] else False \
                   for item in all_number_of_constraints]

    ratios = [minimization_ratio_sus(
                item['to_explorer'], item['to_minimizer'], item['mus']
              ) for item in all_number_of_constraints]

    stratified_ratios = {}
    exactly_mus_instances = []

    for idx, (is_correct, ratio) in enumerate(zip(are_correct, ratios)):
        if ratio < 0.95:
            lower_lim = round(ratio, 1)
            upper_lim = round(lower_lim + 0.1, 1)
            key = f'{lower_lim}-{upper_lim}'
        elif 0.95 <= ratio < 1: key = '0.9-1.0'
        else: key = '=1.0'

        if key == '=1.0':
            exactly_mus_instances.append(all_number_of_constraints[idx]['to_explorer'])

        if key in stratified_ratios:
            stratified_correct, stratified_total = stratified_ratios[key]
            if is_correct:
                stratified_correct += 1
            stratified_total += 1
            stratified_ratios[key] = [stratified_correct, stratified_total]
        else:
            if is_correct:
                stratified_ratios[key] = [1, 1]
            else:
                stratified_ratios[key] = [0, 1]
    
    if '=1.0' not in stratified_ratios: stratified_ratios['=1.0'] = [0, 0]

    stratified_ratios = dict(sorted(stratified_ratios.items()))

    return stratified_ratios, exactly_mus_instances


def extract_number_of_constraints(all_output_files, sc):
    """Extract number of constraints in each input formula, its SUS and MUSes.
    Useful for computing evaluation metrics such as C/N and minimization ratio.

    Arguments:
        all_output_files (list): List of output files containing responses from GPT-x.
        sc (bool): If ``True``, use self-exploration results. Else, use greedy decoding.
    
    Returns:
        number_of_constraints (list): List of dictionaries, each recording the
            number of constraints in the input formula, SUS, and MUS.
    """
    number_of_constraints = []
    for output_file in all_output_files:
        with open(output_file, 'r') as f:
            output_json = json.load(f)

        if 'Number of input clauses' in output_json: key = 'clauses'
        elif 'Number of input constraints' in output_json: key = 'constraints'

        if sc:
            number_of_constraints.append({
                'to_explorer': output_json[f'Number of input {key}'],
                'to_minimizer': output_json['SMT2 Minimizer']['Number of constraints to SMT2-Minimizer'],
                'mus': output_json['SMT2 Minimizer']['Number of constraints in MUS'],
            })
        else:
            _to_minimizer = output_json[f'Number of input {key}']
            if "Validator Response" in output_json["Candidate-0"]:
                if output_json["Candidate-0"]["Validator Response"] == "UNSAT":
                    _to_minimizer = len(output_json["Candidate-0"]["Parsed Subset"])
            number_of_constraints.append({
                'to_explorer': output_json[f'Number of input {key}'],
                'to_minimizer': _to_minimizer,
                'mus': output_json['SMT2 Minimizer']['Number of constraints in MUS'],
            })
    return number_of_constraints



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Intrinsic Evaluation")

    parser.add_argument("--llm", type=str, default='gpt35',
                        choices=['gpt35', 'gpt4', 'geminipro', 'claude'],
                        help="Path to LLM-generated outputs.")
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test'],
                        help=("Evaluation split."))
    parser.add_argument("--sc", action='store_true',
                        help=("Whether to use self-consistency in Explorer LLM."))

    args = parser.parse_args()

    path_to_cache = Path(f'../outputs_all/outputs_{args.llm}') / f'smt2_minimizer/sc/{args.split}'

    all_outputs = list(Path(path_to_cache).iterdir())
    number_of_constraints = extract_number_of_constraints(all_outputs, args.sc)

    success = sum([1 for num in number_of_constraints if num['to_explorer'] > num['to_minimizer']])

    if args.sc: print_msg = 'Printing evaluation statistics for IntelliSMT with Self-Consistency'
    else: print_msg = 'Printing evaluation statistics for IntelliSMT without Self-Consistency'

    print(f"{print_msg}\n{'-'*len(print_msg)}")
    print(f"Number of instances with successful constraint reduction: {success}/{len(number_of_constraints)}")

    import statistics
    pct_min = statistics.mean(
        [100*(item['to_explorer'] - item['to_minimizer'])/item['to_explorer'] \
         for item in number_of_constraints]
    )
    print(f"Mean constraint reduction: {pct_min}%")

    pct_min_corr = statistics.mean(
        [100*(item['to_explorer'] - item['to_minimizer'])/item['to_explorer'] \
         for item in number_of_constraints \
         if item['to_explorer'] != item['to_minimizer']
        ] 
    )
    print(f"Mean constraint reduction, corrected: {pct_min_corr}%")

    results_by_intervals = stratify_by_interval(number_of_constraints)
    print()
    print(f"Stratification based on intervals of total constraints:")
    print(f"Interval\tC/N\tAdjusted r\tAbsolute r")
    for key, ratio_counter in results_by_intervals.items():
        blank_1 = f"{ratio_counter[0][0]}/{ratio_counter[0][1]}"
        try:
            blank_2 = "{:.3f}".format(ratio_counter[1]/ratio_counter[0][0])
        except ZeroDivisionError:
            blank_2 = 0.0
        blank_3 = "{:.3f}".format(ratio_counter[1]/ratio_counter[0][1])
        print(f'{key}\t\t{blank_1}\t{blank_2}\t\t{blank_3}')

    stratified_ratios, exactly_mus_instances = stratify_by_minimization_ratio(number_of_constraints)
    print()
    print(f"Stratification based on reduction ratio:")    
    for key, ratio_counter in stratified_ratios.items():
        print(f'  {key}: {ratio_counter[0]}/{ratio_counter[1]}')
    
    print()
    print(f"Number of cases where we reduce to MUS directly: {stratified_ratios['=1.0']}")
    print(f"  Number of such instances with 0--10 constraints: {len([item for item in exactly_mus_instances if 0 <= item < 10])}")
    print(f"  Number of such instances with 10--20 constraints: {len([item for item in exactly_mus_instances if 10 <= item < 20])}")
    print(f"  Number of such instances with 20--30 constraints: {len([item for item in exactly_mus_instances if 20 <= item < 30])}")
    print(f"  Number of such instances with 30--40 constraints: {len([item for item in exactly_mus_instances if 30 <= item < 40])}")
    print(f"  Number of such instances with 40--50 constraints: {len([item for item in exactly_mus_instances if 40 <= item < 50])}")
