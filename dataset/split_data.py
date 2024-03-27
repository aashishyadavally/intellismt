"""Creates validation and test splits for experiments. Includes stratifications
based on number of constraints, in intervals of 10 constraints.
"""
import argparse
import json
import random
import statistics


SEED = 42


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Create dataset from SMT2 files in the benchmark folder.'
    )
    parser.add_argument('--type', type=str, default='Leetcode',
                        choices=['Leetcode', 'stringfuzz', 'woorpje', 'nornbenchmarks'],
                        help='Choice of benchmark.')
    args = parser.parse_args()

    # Set seed
    random.seed(SEED)

    number_constraints = []

    with open(f'unsat.{args.type}.json', 'r') as f:
        instances = json.load(f)
    
    stratified_instances = {
        '0-10': [], '10-20': [], '20-30': [], '30-40': [], '40-50': []
    }
    mus_times = []
    for instance in instances:
        instance_number_constraints = len(instance['constraints'])
        number_constraints.append(instance_number_constraints)
        if 0 <= instance_number_constraints < 10:
            stratified_instances['0-10'].append(instance)
        elif 10 <= instance_number_constraints < 20:
            stratified_instances['10-20'].append(instance)
        elif 20 <= instance_number_constraints < 30:
            stratified_instances['20-30'].append(instance)
        elif 30 <= instance_number_constraints < 40:
            stratified_instances['30-40'].append(instance)
        elif 40 <= instance_number_constraints < 50:
            stratified_instances['40-50'].append(instance)

        mus_times.append(float(instance['mimimal_unsat_core_time (in ms)']))

    mus_times_mean = statistics.mean(mus_times)

    while True:
        val, test = [], []
        for instances_in_range in stratified_instances.values():
            random.shuffle(instances_in_range)
            n = len(instances_in_range) // 2
            val += instances_in_range[:n]
            test += instances_in_range[n:]

        val_mus_times_mean = statistics.mean([
            float(instance['mimimal_unsat_core_time (in ms)']) for instance in val
        ])
        test_mus_times_mean = statistics.mean([
            float(instance['mimimal_unsat_core_time (in ms)']) for instance in test
        ])
    
        difference = abs(test_mus_times_mean - mus_times_mean)
        if difference / mus_times_mean < 0.01:
            break

    print(f"  Total number of UNSAT instances: {len(instances)}")
    print(f"  Total number of UNSAT instances in validation split: {len(val)}")
    print(f"  Total number of UNSAT instances in test split: {len(test)}")

    with open(f'unsat.{args.type}.val.json', 'w') as f:
        random.shuffle(val)
        json.dump(val, f, indent=2)

    with open(f'unsat.{args.type}.test.json', 'w') as f:
        random.shuffle(test)
        json.dump(test, f, indent=2)
