#!/usr/bin/env python3

"""Perception-style tools for deferring execution and aligning beliefs."""

from typing import TYPE_CHECKING, List, Tuple

from habitat_llm.tools import PerceptionTool
from habitat_llm.utils.grammar import FREE_TEXT

if TYPE_CHECKING:
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface


class _BeliefToolBase(PerceptionTool):
    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.skill_config = skill_config
        self.env_interface: "EnvironmentInterface" = None

    def set_environment(self, env_interface: "EnvironmentInterface") -> None:
        self.env_interface = env_interface

    @property
    def description(self) -> str:
        return self.skill_config.description

    @property
    def argument_types(self) -> List[str]:
        return [FREE_TEXT]


class AskHumanTool(_BeliefToolBase):
    """Tool used when the robot needs to explicitly ask the human."""

    def process_high_level_action(self, input_query, observations) -> Tuple[None, str]:
        super().process_high_level_action(input_query, observations)
        content = input_query or "Requesting clarification."
        if self.env_interface:
            self.env_interface.register_belief_update("AskHuman", content, self.agent_uid)
        return None, f"Asked human: {content}"


class AppendObservationTool(_BeliefToolBase):
    """Tool that appends additional observations to the concept graph."""

    def process_high_level_action(self, input_query, observations) -> Tuple[None, str]:
        super().process_high_level_action(input_query, observations)
        if self.env_interface:
            # mark added observation as confident to encourage progress
            normalized = (input_query or "new observation").strip()
            if normalized:
                self.env_interface.record_concept_confidence(
                    self.agent_uid, {normalized: 1.0}
                )
            self.env_interface.register_belief_update("AppendObservation", normalized, self.agent_uid)
        return None, f"Recorded new observation: {input_query}"


class CorrectHumanTool(_BeliefToolBase):
    """Tool for pushing the robot's belief back to the human."""

    def process_high_level_action(self, input_query, observations) -> Tuple[None, str]:
        super().process_high_level_action(input_query, observations)
        correction = input_query or "Clarifying my belief."
        if self.env_interface:
            self.env_interface.register_belief_update("CorrectHuman", correction, self.agent_uid)
        return None, f"Shared correction with human: {correction}"
