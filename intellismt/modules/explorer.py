"""Contains LLM-based explorer as in IntelliSMT framework (see pipeline.py).
"""
import abc
import os
import time

import google.generativeai as gemini

import torch

from anthropic import Anthropic

from openai import AzureOpenAI

from intellismt.modules.parsers import (
    SubsetParserWithoutAutoCorrection,
    SubsetParserWithEditDistance,
    SubsetParserWithCodeBERTEmbeddings,
)
from intellismt.modules.globals import EXPLORER_SYSTEM, EXPLORER_TEMPLATE
from intellismt.utils import (
    extract_output_constraints,
    OutputFormatError
)


class LLMExplorer(abc.ABC):
    """An LLM-based search-space explorer class.
    """
    @abc.abstractmethod
    def configure_llm(self, name):
        """Configure LLM with Azure OpenAI configurations stored in .env file,
        at the top-level in the repository.
        """
        pass

    def init_system_prompt(self):
        """Initializes system prompt for both Explorer and Minimizer LLMs.

        Returns:
            input_prompt (str): System prompt.
        """
        input_prompt = EXPLORER_SYSTEM.strip()

        return input_prompt

    def init_user_prompt(self):
        """Initializes user prompt for both Explorer and Minimizer LLMs.

        Returns:
            input_prompt (str): User prompt.
        """
        input_prompt = EXPLORER_TEMPLATE.strip()
        return input_prompt

    def apply_zero_shot(self, input_prompt):
        """Initializes input prompt with no examples, as in the zero-shot setting.

        Arguments:
            input_prompt (str): Input prompt.
        
        Returns:
            input_prompt (str): Modified input prompt, as per zero-shot setting.
        """
        input_prompt = input_prompt.replace("\n<EXAMPLES>", "")
        return input_prompt

    def apply_few_shot(self, input_prompt):
        """Initializes input prompt with some examples, as in the few-shot setting.
        These examples are saved in ``intellismt.modules.globals.EXPLORER_EXEMPLARS``
        or ``intellismt.modules.globals.MINIMIZER_EXEMPLARS``.

        Arguments:
            input_prompt (str): Input prompt.
        
        Returns:
            input_prompt (str): Modified input prompt, as per few-shot setting.
        """
        if self.key == "explorer":
            from intellismt.modules.globals import EXPLORER_EXEMPLARS
            exemplars = EXPLORER_EXEMPLARS.strip()
        elif self.key == "minimizer":
            from intellismt.modules.globals import MINIMIZER_EXEMPLARS
            exemplars = MINIMIZER_EXEMPLARS.strip()

        input_prompt = input_prompt.replace("<EXAMPLES>", exemplars)
        return input_prompt

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

    @abc.abstractmethod
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
        pass

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

    def process_response(self, candidate_response, all_constraints):
        """Process LLM's response to extract output subset of constraints.

        Arguments:
            candidate_response (str): Complete GPT-x response, containing all SUSes.
            all_constraints (list): Complete list of constraints in string format.
        
        Returns:
            (tuple): Tuple of GPT response, and extracted subset.
        """
        try:
            subset = extract_output_constraints(candidate_response)
        except OutputFormatError:
            return None

        if hasattr(self, "parser"):
            parsed_subset = self.parser.parse(subset, all_constraints)
            return parsed_subset

        return subset


class GPTExplorer(LLMExplorer):
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

    def configure_llm(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def explore(self, prompt, seed, all_constraints):
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
            candidate_response = response.choices[candidate_id].message.content
            candidate_subset = self.process_response(candidate_response, all_constraints)
            all_candidates.append((candidate_response, candidate_subset, usage))
        return all_candidates


class ClaudeExplorer(LLMExplorer):
    def __init__(
            self, model_name, few_shot, parse_strategy, combine_strategy,
            top_p, temperature, num_responses, max_tokens_to_sample=4096
        ):
        """Explores search space of input constraints to extract unsat core.

        Arguments:
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
        self.configure_llm()
        self.model_name = model_name
        self.few_shot = few_shot
        # Parameters useful to control response generation.
        self.top_p = top_p
        self.max_tokens_to_sample = max_tokens_to_sample
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

    def configure_llm(self):
        self.client = Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'), max_retries=25
        )

    def explore(self, prompt, seed, all_constraints):
        # Currently, Claude does not support multiple responses, and is not
        # directly suitable for self-exploration.
        all_candidates = []
        for _ in range(self.num_responses):
            start_time = time.time()
            response = self.client.messages.create(
                max_tokens=self.max_tokens_to_sample,
                system=prompt[0]['content'],
                messages=prompt[1:],
                model=self.model_name,
                temperature=self.temperature
            )
            end_time = time.time()
            time.sleep(2)

            usage = {
                'completion_tokens': response.usage.output_tokens,
                'prompt_tokens': response.usage.input_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
                'time': end_time - start_time,
            }

            candidate_response = response.content[0].text
            candidate_subset = self.process_response(candidate_response, all_constraints)
            all_candidates.append((candidate_response, candidate_subset, usage))

        return all_candidates


class GeminiExplorer(LLMExplorer):
    def __init__(
            self, few_shot, parse_strategy, combine_strategy, top_p,
            temperature, num_responses,
        ):
        """Explores search space of input constraints to extract unsat core.

        Arguments:
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
        self.configure_llm()
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

    def configure_llm(self):
        gemini.configure(api_key=os.getenv('GEMINI_API_KEY'))

    def explore(self, prompt, seed, all_constraints):
        # Currently, Gemini only supports one response and is not directly suitable
        # for self-exploration.
        all_candidates = []
        for _ in range(self.num_responses):
            start_time = time.time()
            client = gemini.GenerativeModel(
                model_name="gemini-1.5-pro",
                system_instruction=prompt[0]['content'],
            )

            response = client.generate_content(
                prompt[1]['content'],
                generation_config=gemini.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            )
            end_time = time.time()
            time.sleep(5)

            usage = {
                'completion_tokens': response.usage_metadata.candidates_token_count,
                'prompt_tokens': response.usage_metadata.prompt_token_count,
                'total_tokens': response.usage_metadata.total_token_count,
                'time': end_time - start_time,
            }

            candidate_response = '\n'.join([
                part.text for part in response.candidates[0].content.parts
            ])
            candidate_subset = self.process_response(candidate_response, all_constraints)
            all_candidates.append((candidate_response, candidate_subset, usage))

        return all_candidates
