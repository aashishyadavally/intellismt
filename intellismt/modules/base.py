"""Contains :class: ``BaseLLM`` which is the base implementation for LLM-based
state space exploration in ``intellismt.modules.explorer``, or minimization
in ``intellismt.modules.minimizers``.
"""
import os

from openai import AzureOpenAI


class BaseLLM:
    """Base class for LLM-based state space exploration in ``intellismt.modules.explorer``,
    or minimization in ``intellismt.modules.minimizers``.
    """
    def __init__(self):
        self.key = None

    def validate_llm_type(self):
        """Validate use-case for LLM. For now, this class can only be extended to
        'explorer' and 'minimizer' supports.
        """
        if self.key not in ['explorer', 'minimizer']:
            raise ValueError("Invalid LLM type: Only 'explorer' and 'minimizer' are allowed.")

    def configure_llm(self):
        """Configure LLM with Azure OpenAI configurations stored in .env file,
        at the top-level in the repository.
        """
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def init_system_prompt(self):
        """Initializes system prompt for both Explorer and Minimizer LLMs.

        Returns:
            input_prompt (str): System prompt.
        """
        if self.key == "explorer":
            from intellismt.modules.globals import EXPLORER_SYSTEM
            input_prompt = EXPLORER_SYSTEM.strip()
        elif self.key == "minimizer":
            from intellismt.modules.globals import MINIMIZER_SYSTEM
            input_prompt = MINIMIZER_SYSTEM.strip()
        return input_prompt

    def init_user_prompt(self):
        """Initializes user prompt for both Explorer and Minimizer LLMs.

        Returns:
            input_prompt (str): User prompt.
        """
        if self.key == "explorer":
            from intellismt.modules.globals import EXPLORER_TEMPLATE
            input_prompt = EXPLORER_TEMPLATE.strip()
        elif self.key == "minimizer":
            from intellismt.modules.globals import MINIMIZER_TEMPLATE
            input_prompt = MINIMIZER_TEMPLATE.strip()
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
