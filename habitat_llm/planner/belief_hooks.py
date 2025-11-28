#!/usr/bin/env python3

"""
Lightweight helpers for routing planner decisions based on world model
confidence and belief divergence.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BeliefMetrics:
    avg_concept_confidence: float = 1.0
    belief_divergence: float = 0.0
    note: str = ""


def choose_belief_action(decision_conf, metrics: BeliefMetrics) -> Tuple[Optional[str], str]:
    """Return the tool name and a short reason when a hook should run.

    The decision hierarchy is:
    1) If divergence is above the correction threshold, prefer correction.
    2) If average concept confidence is below the configured threshold, add
       more observations.
    3) If divergence is above the warning threshold, ask the human for help.
    """

    if decision_conf is None:
        return None, ""

    div_threshold = decision_conf.get("divergence_threshold", 0.3)
    correction_threshold = decision_conf.get("correction_divergence_threshold", div_threshold * 1.5)
    confidence_threshold = decision_conf.get("concept_confidence_threshold", 0.5)

    if metrics.belief_divergence >= correction_threshold:
        action = decision_conf.get("correction_action", "CorrectHuman")
        reason = (
            f"Belief divergence {metrics.belief_divergence:.2f} exceeds correction "
            f"threshold {correction_threshold:.2f}."
        )
        return action, reason

    if metrics.avg_concept_confidence < confidence_threshold:
        action = decision_conf.get("low_confidence_action", "AppendObservation")
        reason = (
            f"Average concept confidence {metrics.avg_concept_confidence:.2f} is below "
            f"threshold {confidence_threshold:.2f}."
        )
        return action, reason

    if metrics.belief_divergence >= div_threshold:
        action = decision_conf.get("high_divergence_action", "AskHuman")
        reason = (
            f"Belief divergence {metrics.belief_divergence:.2f} exceeds "
            f"threshold {div_threshold:.2f}."
        )
        return action, reason

    return None, metrics.note
