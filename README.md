# Artifact for "Large Language Models for Safe Minimization"

``IntelliSMT`` is a learning-aided solver to safely minimize input string constraint systems, useful for a parallelized, partial enumeration of its minimal unsatifiable subsets (MUSes). The source code, data, and model outputs are publicly available on GitHub (https://github.com/se-doubleblind/intellismt).

## Getting Started
This section describes the preqrequisites, and contains instructions, to get the project up and running.

### Setup 

#### Hardware Requirements
``intellismt`` requires access to OpenAI API credentials to use GPT-3.5 and GPT-4. However, we also provide all GPT responses for the dataset, and this can be skipped for experiments' replication.

#### Project Environment
Currently, ``intellismt`` works well on Ubuntu OS, and can be set up easily with all the prerequisite packages by following these instructions (if ``conda`` is already installed, update to the latest version with ``conda update conda``, and skip steps 1 - 3):
  1. Download the latest, appropriate version of [conda](https://repo.anaconda.com/miniconda/) for your machine (tested with ``conda 23.11.0``).
  2. Install  it by running the `conda_install.sh` file, with the command:
     ```bash
     $ bash conda_install.sh
     ```
  3. Add `conda` to bash profile:
     ```bash
     $ source ~/.bashrc
     ```
  4. Navigate to ``intellismt`` (top-level directory) and create a conda virtual environment with the included `environment.yml` file using the following command:     
     ```bash
     $ conda env create -f environment.yml
     ```

     To test successful installation, make sure ``intellismt`` appears in the list of conda environments returned with ``conda env list``.
  5. Activate the virtual environment with the following command:     
     ```bash
     $ conda activate intellismt
     ```

### Directory Structure

#### 1. Data Artifacts
Navigate to ``intellismt/dataset/`` to find:
* the processed string benchmark files (``unsat.LeetCode.{val|test}.json``) -- use these files to replicate experiments results in the paper from scratch.
* ``sampled_qualitative.json`` -- sampled dataset from test split for qualitative analysis.

#### 2. Code
* ``intellismt``: package code used in ``pipeline.py``
* ``experiments``: Code for all experiments (RQ1-RQ3)

#### 3. Qualitative Analysis
Navigate to ``intellismt/qualitative-analysis/`` to find the details about the empirical study (see RQ4) in the paper.

### Usage Guide
1. Navigate to ``experiments/`` to find the source code for replicating the experiments in RQ1-RQ3 in the paper. This assumes the LLM outputs (e.g., ``outputs_gpt35``, as stored in ``outputs_all``) are being used.

  * **Option 1**. Run all experiments to print Tables II, III, and overlapping analysis in Section VIII:
  ```bash
  python all_illustrations.py --path_to_data ../dataset --split test
  ```
  
  * **Option 2.** Run experiments independently:
  
    - Intrinsic evaluation (RQ1 + RQ2)
      * *Run GPT models*
      ```bash
      python intrinsic.py --path_to_cache ../outputs_all/outputs_gpt4 --split test --sc
      ```
  
      * *Run naive baseline*
      ```bash
      python intrinsic_naive.py --path_to_data ../dataset --split test
      ```
  
    - Extrinsic evaluation (RQ3)
    ```bash
    python extrinsic.py --path_to_data ../dataset --path_to_cache ../outputs_all/outputs_gpt4  --split test
    ```

2. Navigate to top-level directory to build GPT-x model outputs from scratch.

    **Option 3.** Run ``pipeline.py``, which is the main entry-point for the package. It has the following arguments:

    | Argument                | Default                 | Description |
    | :---------------------: | :---------------------: | :---- |
    | ``--path_to_data``      | ``../dataset``          | Path to processed string constraints dataset file  |
    | ``--path_to_outputs``   | ``../outputs_all/outputs_gpt4``     | Path to cache GPT-x responses |
    | ``--exrange``           | ``0 5``                 | Range of examples to process: upper-limit not considered |
    | ``--benchmark``         | ``Leetcode``            | Benchmark dataset |
    | ``--few_shot``          | ``False``               | Whether to use exemplars for few-shot learning or not |
    | ``--explore_only``      | ``False``               | If True, run only Stage 1 and Stage 2 in IntelliSMT pipeline |
    | ``--minimize_only``     | ``False``               | If True, run only Stage 3 in IntelliSMT pipeline |
    | ``--num_responses``     | ``5``                   | Number of responses to generate for Explorer LLM |
    | ``--seed``              | ``42``                  | Set common system-level random seed |
    | ``--top_p``             | ``0.7``                 | Only the most probable tokens with probabilities that add up to top_p or higher are considered during decoding |
    | ``--temperature``       | ``0.6``                 | A value used to warp next-token probabilities in sampling mode |
    | ``--use_minimizer_llm`` | ``False``               | Whether to use ``intellismt.modules.minimizers.LLMMinimizer`` for finding MUS |
    | ``--split``             | ``test``                | Evaluation split |
    | ``--parse_strategy``    | ``edit``                | Strategy to parse and validate unsatisfiable subset returned by Explorer LLM |
    | ``--combine_strategy``  | ``mean`                 | Strategy to aggregate sub-token embeddings. Only valid when ``--parse_strategy`` is set to ``embedding`` |
    | ``--timeout``           | ``30000``               | Timeout for SMT-2 validator (in milliseconds) |

    ***Sample usage:***
    ```bash
    python pipeline.py --path_to_data ../dataset --path_to_outputs ../outputs_all/outputs_gpt4
    ```

    ***Note:*** We use the ``seed`` parameter when using the OpenAI chat completion client to ensure reproducible outputs. However, as the OpenAI team notes, sometimes, determinism may be impacted due to necessary changes to model configurations ([[link]](https://platform.openai.com/docs/guides/text-generation/reproducible-outputs)).

    ***Note:*** A prerequisite to running ``pipeline.py`` is that it expects an ``.env`` file at the top-level directory with the following key-value pairs:
    * AZURE_OPENAI_ENDPOINT=<name-of-endpoint>
    * OPENAI_API_KEY=<openai-api-key>
    * OPENAI_API_VERSION=<api-version>
    * AZURE_OPENAI_DEPLOYMENT_NAME=<gpt-model-deployment-name>
    * ANTHROPIC_API_KEY=<anthropic-api-key>
    * GEMINI_API_KEY=<gemini-api-key>

## Contributing Guidelines
There are no specific guidelines for contributing, apart from a few general guidelines we tried to follow, such as:
* Code should follow PEP8 standards as closely as possible
* Code should carry appropriate comments, wherever necessary, and follow the docstring convention in the repository.

If you see something that could be improved, send a pull request! 
We are always happy to look at improvements, to ensure that `intellismt`, as a project, is the best version of itself. 

If you think something should be done differently (or is just-plain-broken), please create an issue.

## License
See the [LICENSE](https://github.com/se-doubleblind/intellismt/tree/main/LICENSE) file for more details.
