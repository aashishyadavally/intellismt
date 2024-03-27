"""Contains LLM-based explorer as in IntelliSMT framework (see pipeline.py).
"""
import time
import torch

from intellismt.modules.base import BaseLLM
from intellismt.modules.parsers import (
    SubsetParserWithoutAutoCorrection,
    SubsetParserWithEditDistance,
    SubsetParserWithCodeBERTEmbeddings,
)
from intellismt.utils import (
    extract_output_constraints,
    OutputFormatError
)


class LLMExplorer(BaseLLM):
    """An LLM-based search-space explorer class.
    """
    def __init__(
            self, model_name, few_shot, parse_strategy, combine_strategy,
            top_p, temperature, num_responses,
        ):
        """Explores search space of input constraints to extract unsat core.

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
        self.key = 'explorer'
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
            formula += f"Constraint {idx+1}: <c>{clause}</c>,\n"
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

    def process_response(self, response, all_constraints, candidate_id):
        """Process GPT-x's response to extract output subset of constraints.

        Arguments:
            response (str): Complete GPT-x response, containing all SUSes.
            all_constraints (list): Complete list of constraints in string format.
            candidate_id (int): Identifier for candidate SUS.
        
        Returns:
            (tuple): Tuple of GPT response, and extracted subset.
        """
        candidate_response = response.choices[candidate_id].message.content
        try:
            subset = extract_output_constraints(candidate_response)
        except OutputFormatError:
            return candidate_response, None

        if hasattr(self, "parser"):
            parsed_subset = self.parser.parse(subset, all_constraints)
            return candidate_response, parsed_subset

        return candidate_response, subset

    def explore(self, prompt, seed, all_constraints):
        """Explore input constraint space, ensuring replicability with
        ``seed`` to OpenAI client.

        Arguments:
            prompt (str): Input prompt.
            seed (int): Seed to OpenAI client.
            all_constraints (list): Complete list of constraints in string format.
        
        Returns:
            all_candidates (list): List of tuples containing all candidate responses,
            extracted subsets (i.e., SUSes), and their API usage statistics.
        """
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            seed=seed,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.num_responses,
            timeout = 240
        )
        end_time = time.time()
        usage = {
            'completion_tokens': response.usage.completion_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
            'time': end_time - start_time,
        }
        all_candidates = []
        for candidate_id in range(self.num_responses):
            candidate_response, candidate_subset = self.process_response(
                response, all_constraints, candidate_id
            )
            all_candidates.append((candidate_response, candidate_subset, usage))
        return all_candidates

    def explore_without_self_consistency(self, prompt, seed, all_constraints):
        """Explore input constraint space without self-consistency, i.e., choose
        candidate SUS greedily.

        Arguments:
            prompt (str): Input prompt.
            seed (int): Seed to OpenAI client.
            all_constraints (list): Complete list of constraints in string format.
        
        Returns:
            (tuple): Tuple containing candidate response, extracted subsets (i.e., SUS),
            and its API usage statistics.
        """
        if self.num_responses != 1:
            raise AssertionError('Only valid if ``num_responses`` is set to 1.')

        all_candidates = self.explore(prompt, seed, all_constraints)
        response, subset, usage = all_candidates[0]
        return response, subset, usage

    def explore_with_self_consistency(self, prompt, seed, all_constraints):
        """Explore input constraint space with self-consistency, i.e., return
        ``num_responses`` candidate SUSes.

        Arguments:
            prompt (str): Input prompt.
            seed (int): Seed to OpenAI client.
            all_constraints (list): Complete list of constraints in string format.
        
        Returns:
            (list): List of tuples containing all candidate responses, extracted
            subsets (i.e., SUSes), and their API usage statistics.
        """
        return self.explore(prompt, seed, all_constraints)
