"""Contains different parser options for IntelliSMT framework to mitigate
orthographic hallucinations in GPT-x's response.
"""
import os
import tempfile
from abc import ABC

from simpletransformers.language_representation import RepresentationModel

import torch


class BaseParser(ABC):
    """The base class for a parser with :func: ``parse``, which parses a set
    of constraints output by GPT-x model, to return the orthographically-correct subset.
    """
    def parse(self, subset: list, constraints:list) -> list:
        """Parses a given constraint subset in string format to retrieve corresponding
        orthographically-correct subset.

        Arguments:
            subset (list): Constraint subset from GPT-x's response in string format.
            constraints (list): Complete list of constraints in string format.
        """
        pass


class SubsetParserWithoutAutoCorrection(BaseParser):
    """A parser class that does not mitigate the orthographic hallucinations
    in GPT-x's response. Thus, output ``subset`` can have constraints that do
    not belong to the input constraint system.

    Arguments:
        subset (list): Constraint subset from GPT-x's response in string format.
        constraints (list): Complete list of constraints in string format.

    Returns:
        subset (list): Constraint subset from GPT-x's response in string format.
    """
    def parse(self, subset, constraints):
        """Skips parsing. Returns input ``subset``.

        Arguments:
            subset (list): Constraint subset from GPT-x's response in string format.
            constraints (list): Complete list of constraints in string format.
        """
        return subset


class SubsetParserWithEditDistance(BaseParser):
    """A parser class that mitigates the orthographic hallucinations in GPT-x's
    response based on Levenshtein Distance.

    Arguments:
        subset (list): Constraint subset from GPT-x's response in string format.
        constraints (list): Complete list of constraints in string format.

    Returns:
        subset (list): Constraint subset from GPT-x's response in string format.
    """
    def levenshtein_distance(self, s1, s2):
        """Quick implementation for computing Levenshtein Distance between two strings.        
        Adapted from [1].

        Arguments:
            s1 (str): First string
            s2 (str): Other string
        
        Returns:
            (float): Levenshtein distance between ``s1`` and ``s2``.

        References:
        [1] https://stackoverflow.com/questions/2460177/edit-distance-in-python
        """
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def parse(self, subset, constraints):
        """Parses a given constraint subset in string format to retrieve corresponding
        orthographically-correct subset based on Levenshtein Distance.

        Arguments:
            subset (list): Constraint subset from GPT-x's response in string format.
            constraints (list): Complete list of constraints in string format.
        
        Returns:
            parsed_constraints (list): Parsed constraint subset.
        """
        parsed_constraints = []
        for output_constraint in subset:
            if output_constraint in constraints and output_constraint not in parsed_constraints:
                parsed_constraints.append(output_constraint)
            else:
                # Retrieve constraint corresponding to closest embedding.
                distances = [self.levenshtein_distance(output_constraint, constraint) \
                             for constraint in constraints]
                index_to_most_similar_constraint = distances.index(min(distances))
                corrected_constraint = constraints[index_to_most_similar_constraint]
                if corrected_constraint not in parsed_constraints:
                    parsed_constraints.append(corrected_constraint)
        return parsed_constraints


class SubsetParserWithCodeBERTEmbeddings(BaseParser):
    """A parser class that mitigates the orthographic hallucinations in GPT-x's
    response based on cosine similarity between CodeBERT embeddings.

    Implementation from ``simpletransformers.language_representation`` used
    for building CodeBERT embeddings for the constraints.

    Arguments:
        subset (list): Constraint subset from GPT-x's response in string format.
        constraints (list): Complete list of constraints in string format.

    Returns:
        subset (list): Constraint subset from GPT-x's response in string format.
    """

    def __init__(self, combine_strategy, use_cuda):
        """Initializes :class: ``intellismt.modules.parsers.SubsetParserWithCodeBERTEmbeddings``.

        Arguments:
            combine_strategy (str): One of "mean" and "concat", used to aggregate/combine
                sub-token embeddings in a constraint.
            use_cuda (bool): If True, use
        """
        self.model = RepresentationModel(
            "roberta",
            "microsoft/codebert-base",
            use_cuda=use_cuda,
        )
        self.combine_strategy = combine_strategy
        self.cache_dir = tempfile.TemporaryDirectory()
        self.constraint_embeddings = None

    def retrieve_constraint_embeddings(self, constraints):
        """Load embeddings for constraints. If they have not been created and cached
        yet, do so with CodeBERT PLM. If not, retrieve from cache.

        Arguments:
            constraints (list): Complete list of constraints in string format.
        
        Returns:
            constraint_embeddings (list): List of CodeBERT embeddings corresponding
                to the constraints.
        """
        if not os.listdir(self.cache_dir):
            # If constraint embeddings have not been created and cached yet, do so.
            constraint_embeddings = self.encode_constraints(constraints)
            self.cache_embeddings(constraint_embeddings)
        else:
            # Retrieve cached constraint embeddings.
            constraint_embeddings = self.load_embeddings_from_cache(len(constraints))
        return constraint_embeddings

    def encode_constraints(self, constraints):
        """Created and cache embeddings for constraints with CodeBERT PLM.

        Arguments:
            constraints (list): Complete list of constraints in string format.
        
        Returns:
            constraint_embeddings (list): List of CodeBERT embeddings corresponding
                to the constraints.
        """
        constraint_embeddings = self.model.encode_sentences(
                                constraints,
                                combine_strategy=self.combine_strategy
                            )
        return constraint_embeddings

    def cache_embeddings(self, constraint_embeddings):
        """Cache constraint embeddings in local store for quick reference.

        Arguments:
            constraint_embeddings (list): List of constraint embeddings extracted
                from CodeBERT PLM.
        """
        for idx, tensor in enumerate(constraint_embeddings):
            torch.save(tensor, f"{self.cache_dir}/{idx}.constraint.pt")

    def load_embeddings_from_cache(self, N):
        """Load constraint embeddings from local cache.

        Arguments:
            N (int): Numver of constraint embeddings to load.
        """
        constraint_embeddings = []
        for idx in range(N):
            constraint_embeddings.append(torch.load(f"{self.cache_dir}/{idx}.constraint.pt"))
        return constraint_embeddings

    def cleanup(self):
        """Clean up constraint embeddings' cache.
        """
        self.cache_dir.cleanup()

    def parse(self, subset, constraints):
        """Parses a given constraint subset in string format to retrieve corresponding
        orthographically-correct subset based on cosine similarity distance from
        CodeBERT embeddings.

        Arguments:
            subset (list): Constraint subset from GPT-x's response in string format.
            constraints (list): Complete list of constraints in string format.
        
        Returns:
            parsed_constraints (list): Parsed constraint subset.
        """
        parsed_constraints = []
        for output_constraint in subset:
            if output_constraint in constraints: parsed_constraints.append(output_constraint)
            else:
                if not self.constraint_embeddings:
                    constraint_embeddings = self.retrieve_constraint_embeddings(constraints)
                    setattr(self, "constraint_embeddings", constraint_embeddings)
                
                # Retrieve constraint corresponding to closest embedding.
                distances = [torch.nn.functional.cosine_similarity(output_constraint, constraint) \
                             for constraint in self.constraint_embeddings]
                index_to_most_similar_constraint = distances.index(max(distances))
                corrected_constraint = constraints[index_to_most_similar_constraint]
                parsed_constraints.append(corrected_constraint)
        return parsed_constraints
