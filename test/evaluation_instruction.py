from typing import List, TypedDict


class EvaluationInstruction(TypedDict):
    start: List[float]
    end: List[float]
    step: List[float]
