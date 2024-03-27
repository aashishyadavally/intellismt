"""Contains SMT-based and LLM-based minimizers, as used in Step-3
in IntelliSMT framework (see pipeline.py).
"""
import time

import torch

import cvc5

from intellismt.modules.base import BaseLLM
from intellismt.modules.parsers import (
    SubsetParserWithoutAutoCorrection,
    SubsetParserWithEditDistance,
    SubsetParserWithCodeBERTEmbeddings,
)
from intellismt.utils import (
    OutputFormatError,
    build_smt2_formula_from_string_constraints,
    extract_output_constraints,
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


class LLMMinimizer(BaseLLM):
    """An LLM-based minimizer class.
    """
    def __init__(
            self, model_name, few_shot, parse_strategy, combine_strategy,
            top_p, temperature, num_responses,
        ):
        """Initializes :class: ``LLMMinimizer``.

        Arguments:
            model_name (str): Azure OpenAI deployment name of GPT-x model.
            few_shot (bool): If ``True``, use exemplars for few-shot setting.
            parse_strategy (str): One of 'edit' and 'embedding'. If 'edit', uses Levenshtein
                Distance for parsing. If 'embedding', uses CodeBERT embeddings-based cosine
                similarity distances for parsing.
            combine_strategy (str): Combining strategy to aggregate sub-token embeddings.
                Only valid when ``parse_strategy`` is ``embedding`
            top_p (float): Only the most probable tokens with probabilities that add up to
                ``top_p`` or higher are considered during decoding. The valid range is 0.0
                to 1.0. 1.0 is equivalent to disabled and is the default. Only applies to
                sampling mode.
            temperature (float): A value used to warp next-token probabilities in sampling mode.
            num_responses (int): Number of responses to generate for Explorer LLM. Useful for
                experimenting with self-exploration.
        """
        super().__init__()
        self.key = 'minimizer'
        self.validate_llm_type()
        self.configure_llm()
        # The model name you chose for deploying the GPT-X model.
        self.model_name = model_name
        self.few_shot = few_shot
        # Parameters useful to control response generation.
        self.top_p = top_p
        self.temperature = temperature
        # Control number of generated responses.
        self.num_responses = num_responses

        if parse_strategy == 'edit':
            self.parser = SubsetParserWithEditDistance()
        elif parse_strategy == 'embeddings':
            if torch.cuda.is_available(): use_cuda = True
            else: use_cuda = False

            self.parser = SubsetParserWithCodeBERTEmbeddings(
                combine_strategy=combine_strategy,
                use_cuda=use_cuda,
            )
        else:
            self.parser = SubsetParserWithoutAutoCorrection()

    def apply_formula(self, prompt, formula):
        """Add input formula to prompt.

        Arguments:
            prompt (str): Input prompt.
            formula (str): Input formula.
        
        Returns:
            prompt (str): Modified formula.
        """
        prompt = prompt.replace("<FORMULA>", formula)
        return prompt

    def get_prompt(self, all_constraints):
        """Get initial prompt for Minimizer LLM.

        Arguments:
            all_constraints (list): Complete list of constraints in string format.

        Returns:
            prompt (dict): Input prompt to GPT-x.
        """
        formula = "[\n"
        for idx, clause in enumerate(all_constraints):
            formula += f"{clause} ,\n"
        formula += "]"

        input_prompt = self.init_user_prompt()
        input_prompt = self.apply_formula(input_prompt, formula)

        if self.few_shot:
            input_prompt = self.apply_few_shot(input_prompt)
        else:
            input_prompt = self.apply_zero_shot(input_prompt)

        prompt = [
            {"role": "system", "content": self.init_system_prompt()},
            {"role": "user", "content": input_prompt},
        ]
        return prompt

    def minimize(self, prompt, seed, all_constraints):
        """Minimize input constraints, as in ``prompt``, ensuring replicability
        with ``seed`` to OpenAI client.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            seed=seed,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.num_responses,
            timeout = 120
        )
        self.usage = response.usage
        response = response.choices[0].message.content
        try:
            subset = extract_output_constraints(response)
        except OutputFormatError:
            return response, None

        if hasattr(self, "parser"):
            parsed_subset = self.parser.parse(subset, all_constraints)
            return response, parsed_subset

        return response, subset
