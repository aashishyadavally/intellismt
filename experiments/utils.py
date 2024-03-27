"""Utilities for ``experiments`` module.

Disclaimer: Might repeat some utilities in ``intellismt.modules.utils``.
Need to refactor, hopefully sometime soon.
"""
import cvc5


def build_smt2_formula_from_string_constraints(
        constraints, all_constraints, all_smt2_constraints, placeholder
    ):
    """Builds SMT2-Lib formula from constraints' subset, retrieved from GPT-x's response.

    Arguments:
        constraints (list): Constraint subset from GPT-x's response in string format.
        all_constraints (list): Complete list of constraints in string format.
        all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
            There is one-to-one correspondence with constraints in ``all_constraints``.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
    
    Returns:
        smt2_formula (str): SMT2-Lib input format string, where assertions corresponding
            to ``constraints`` are inserted in ``placeholder``.
    """
    constraints_idx = [all_constraints.index(constraint) for constraint in constraints]
    smt2_assertions = [f"(assert {all_smt2_constraints[idx]})" for idx in constraints_idx]
    assertions = '\n'.join(smt2_assertions)
    smt2_formula = placeholder.replace("<ASSERT>", assertions, 1)
    return smt2_formula


def set_cvc5_options_for_unsat(solver):
    """Initialize configuration for ``cvc5.Solver``, to prove unsatisfiability.

    Arguments:
        solver (cvc5.Solver): SMT solver.
    """
    solver.setOption("print-success", "true")
    solver.setOption("produce-models", "true")
    return


def set_cvc5_options_for_mus(solver):
    """Initialize configuration for ``cvc5.Solver``.
    If ``optimize_for_mus`` is ``True``, option ``minimal-unsat-cores`` is 
    set to ``true``. This gets the solver to look for MUSes.

    Arguments:
        solver (cvc5.Solver): SMT solver.
        optimize_for_mus (bool): If ``True``, initialize for solving MUSes.
    """
    solver.setOption("print-success", "true")
    solver.setOption("produce-models", "true")
    solver.setOption("produce-unsat-cores", "true")
    solver.setOption("unsat-cores-mode", "assumptions")
    solver.setOption("produce-difficulty", "true")
    solver.setOption("minimal-unsat-cores", "true")
    return


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
