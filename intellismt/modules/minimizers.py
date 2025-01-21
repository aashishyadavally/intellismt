"""Contains SMT-based and LLM-based minimizers, as used in Step-3
in IntelliSMT framework (see pipeline.py).
"""
import time

import cvc5

from intellismt.utils import (
    build_smt2_formula_from_string_constraints,
    parse_input_formula,
    set_cvc5_options_for_mus,
)


class SMT2Minimizer:
    """An SMT-based minimizer class.
    """
    def minimize(
            self, unsat_core, all_constraints, all_smt2_constraints, all_cvc5_assertions,
            placeholder
    ):
        """Minimizes given input subset of constraints (i.e., unsat core), to extract
        its minimal unsatisfiable subset (MUS).

        Arguments:
            unsat_core (list): Input subset of constraints in string format.
            all_constraints (list): Complete list of constraints in string format.
            all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
                There is one-to-one correspondence with constraints in ``all_constraints``.
            placeholder (str): SMT2-Lib input file string placeholder, where all assertions
                are represented by "<ASSERT>" keyword.
        
        Returns:
            (dict): MUS, and corresponding statistics.

        """
        smt2_formula = build_smt2_formula_from_string_constraints(
            unsat_core, all_constraints, all_smt2_constraints, placeholder
        )

        solver = cvc5.Solver()
        if 'set-logic' not in smt2_formula:
            solver.setLogic("QF_SLIA")
        set_cvc5_options_for_mus(solver, True)
        parse_input_formula(solver, smt2_formula, "smt_formula")

        mus_start_time = time.time()
        result = solver.checkSat()
        assert result.isUnsat

        mus = solver.getUnsatCore()
        mus_end_time = time.time()
        mus_statistics = solver.getStatistics().get()

        mus_constraints = [all_constraints[all_cvc5_assertions.index(str(constraint))] \
                            for constraint in mus]

        return {
            'cvc5_MUS_assertions': [str(constraint) for constraint in mus],
            'MUS': mus_constraints,
            'MUS_time (in ms)': "{:.3f}".format((mus_end_time - mus_start_time)*1000),
            'MUS_statistics': mus_statistics,
        }
