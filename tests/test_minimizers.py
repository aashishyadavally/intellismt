"""Contains tests for ``intellismt.modules.minimizers.SMT2Minimizer``.
"""
import json
import random
from pathlib import Path

from intellismt.modules.minimizers import SMT2Minimizer


SEED = 42


def test_smt2_minimizer():
    unsat_files = [file for file in list((Path.cwd() / 'dataset').iterdir()) \
                   if file.name.startswith('unsat')]
    random.seed(SEED)
    unsat_file = random.choice(unsat_files)
    with open(unsat_file, 'r') as f:
        instances = json.load(f)

    test_instance = random.choice(instances)

    smt2_minimizer = SMT2Minimizer()
    mus = smt2_minimizer.minimize(
         test_instance['constraints'], 
         test_instance['constraints'],
         test_instance['smt2_constraints'],
         test_instance['cvc5_assertions'],
         test_instance['smt2_formula_placeholder'],
    )
    print(mus['cvc5_MUS_assertions'])
    print(test_instance['minimal_unsat_core'])

    # Assert correct MUS is extracted from input formula.
    assert mus['cvc5_MUS_assertions'] == test_instance['minimal_unsat_core']

    # Assert mapping from CVC5 assertion to Z3 constraint is maintaned.
    for mapped_constraint in mus['MUS']:
        assert mapped_constraint in test_instance['constraints']
