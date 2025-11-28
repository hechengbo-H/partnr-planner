"""Demonstration of belief-aware defer/ask/correct actions without Habitat."""

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from habitat_llm.planner.belief_hooks import BeliefMetrics, choose_belief_action
from habitat_llm.tools.perception.belief_tools import (
    AppendObservationTool,
    AskHumanTool,
    CorrectHumanTool,
)
from habitat_llm.world_model.world_graph import WorldGraph


class MiniEnvInterface:
    def __init__(self):
        self.robot_agent_uid = 0
        self.human_agent_uid = 1
        self.world_graph = {
            0: WorldGraph(),
            1: WorldGraph(),
        }
        self.belief_divergence = 0.5
        self.belief_update_log = []

    def recompute_belief_divergence(self):
        return self.belief_divergence

    def record_concept_confidence(self, agent_uid, updates):
        self.world_graph[agent_uid].concept_confidence.update(updates)

    def register_belief_update(self, action, content, agent_uid):
        self.belief_update_log.append((action, content, agent_uid))


def _make_tool(cls, name: str, env):
    tool = cls(SimpleNamespace(name=name, description=f"Demo tool {name}"))
    tool.set_environment(env)
    return tool


def run_demo():
    conf_path = Path(__file__).resolve().parents[1] / "conf" / "examples" / "belief_hook_demo.yaml"
    decision_conf = OmegaConf.load(conf_path)["demo"]["decision_hooks"]
    env = MiniEnvInterface()
    env.world_graph[0].concept_confidence = {"apple": 0.3}
    metrics = BeliefMetrics(
        avg_concept_confidence=env.world_graph[0].average_concept_confidence(),
        belief_divergence=env.belief_divergence,
    )

    action, reason = choose_belief_action(decision_conf, metrics)
    print(f"Selected hook action: {action} because {reason}")

    tool_map = {
        "AppendObservation": _make_tool(AppendObservationTool, "AppendObservation", env),
        "AskHuman": _make_tool(AskHumanTool, "AskHuman", env),
        "CorrectHuman": _make_tool(CorrectHumanTool, "CorrectHuman", env),
    }
    _, response = tool_map[action].process_high_level_action("apple on table?", {})  # type: ignore
    print(f"Tool response: {response}")
    print(f"Belief log: {env.belief_update_log}")


if __name__ == "__main__":
    run_demo()
