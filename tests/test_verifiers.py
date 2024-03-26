"""Contains tests for ``intellismt.modules.verifiers.UNSATVerifier``
and ``intellismt.modules.verifiers.MUSVerifier.
"""
import json
import random
from pathlib import Path

from intellismt.modules.verifiers import UNSATVerifier, MUSVerifier


SEED = 42


def test_unsat_verifier():
    unsat_files = [file for file in list((Path.cwd() / 'dataset').iterdir()) \
                   if file.name.startswith('unsat')]
    random.seed(SEED)
    unsat_file = random.choice(unsat_files)
    with open(unsat_file, 'r') as f:
        instances = json.load(f)

    test_instance = random.choice(instances)

    unsat_verifier = UNSATVerifier()
    result = unsat_verifier.check(
        test_instance['constraints'], 
        test_instance['constraints'],
        test_instance['smt2_constraints'],
        test_instance['smt2_formula_placeholder'],
    )
    # Assert result of an unsatisfiable formula is UNSAT.
    assert result == True


def test_mus_verifier():
    unsat_files = [file for file in list((Path.cwd() / 'dataset').iterdir()) \
                   if file.name.startswith('unsat')]
    random.seed(SEED)
    unsat_file = random.choice(unsat_files)
    with open(unsat_file, 'r') as f:
        instances = json.load(f)

    test_instance = random.choice(instances)

    mus_verifier = MUSVerifier()
    result_formula = mus_verifier.check(
        test_instance['constraints'], 
        test_instance['constraints'],
        test_instance['smt2_constraints'],
        test_instance['smt2_formula_placeholder'],
    )
    # For the entire formula, assert that this is not the MUS.
    assert result_formula == False

    mus_verifier.reset()

    constraints_idx = [test_instance['cvc5_assertions'].index(constraint) \
                       for constraint in test_instance['minimal_unsat_core']]
    mus_constraints = [test_instance['constraints'][idx] for idx in constraints_idx]

    result_mus = mus_verifier.check(
        mus_constraints,
        test_instance['constraints'],
        test_instance['smt2_constraints'],
        test_instance['smt2_formula_placeholder'],
    )
    # For the MUS, assert that this is indeed the MUS.
    assert result_mus == True
