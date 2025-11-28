import importlib.util
import pathlib
import types
import sys

_belief_hooks_path = pathlib.Path(__file__).resolve().parents[1] / "planner" / "belief_hooks.py"
_belief_spec = importlib.util.spec_from_file_location("belief_hooks", _belief_hooks_path)
belief_hooks = importlib.util.module_from_spec(_belief_spec)
assert _belief_spec and _belief_spec.loader
_belief_spec.loader.exec_module(belief_hooks)  # type: ignore
BeliefMetrics = belief_hooks.BeliefMetrics
choose_belief_action = belief_hooks.choose_belief_action

# Stub minimal utils.grammar to avoid heavy habitat dependency when importing tools
stub_utils = types.ModuleType("habitat_llm.utils")
stub_grammar = types.ModuleType("habitat_llm.utils.grammar")
stub_grammar.FREE_TEXT = "free_text"
stub_utils.grammar = stub_grammar
sys.modules["habitat_llm.utils"] = stub_utils
sys.modules["habitat_llm.utils.grammar"] = stub_grammar

_belief_tools_path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "perception" / "belief_tools.py"
_belief_tools_spec = importlib.util.spec_from_file_location("belief_tools", _belief_tools_path)
belief_tools = importlib.util.module_from_spec(_belief_tools_spec)
assert _belief_tools_spec and _belief_tools_spec.loader
_belief_tools_spec.loader.exec_module(belief_tools)  # type: ignore
AppendObservationTool = belief_tools.AppendObservationTool
AskHumanTool = belief_tools.AskHumanTool


class _DummyGraph:
    def __init__(self):
        self.concept_confidence = {}


class _MiniEnv:
    def __init__(self):
        self.robot_agent_uid = 0
        self.human_agent_uid = 1
        self.world_graph = {0: _DummyGraph(), 1: _DummyGraph()}
        self.world_graph[0].concept_confidence = {"cup": 0.2}
        self.belief_divergence = 0.4
        self.belief_update_log = []

    def register_belief_update(self, action, content, agent_uid):
        self.belief_update_log.append((action, content, agent_uid))

    def record_concept_confidence(self, agent_uid, updates):
        self.world_graph[agent_uid].concept_confidence.update(updates)

    def recompute_belief_divergence(self):
        return self.belief_divergence


def _make_tool(cls, name: str, env: _MiniEnv):
    tool = cls(types.SimpleNamespace(name=name, description=name))
    tool.set_environment(env)
    return tool


def test_choose_belief_action_prefers_correction():
    decision_conf = {
        "enable": True,
        "divergence_threshold": 0.2,
        "correction_divergence_threshold": 0.3,
        "concept_confidence_threshold": 0.5,
    }
    metrics = BeliefMetrics(avg_concept_confidence=0.9, belief_divergence=0.5)
    action, reason = choose_belief_action(decision_conf, metrics)
    assert action == "CorrectHuman"
    assert "0.5" in reason


def test_append_observation_tool_updates_confidence():
    env = _MiniEnv()
    tool = _make_tool(AppendObservationTool, "AppendObservation", env)
    _, response = tool.process_high_level_action("mug on table", {})
    assert "mug on table" in env.world_graph[0].concept_confidence
    assert env.world_graph[0].concept_confidence["mug on table"] == 1.0
    assert env.belief_update_log
    assert "Recorded new observation" in response


def test_ask_human_tool_logs_feedback():
    env = _MiniEnv()
    tool = _make_tool(AskHumanTool, "AskHuman", env)
    _, response = tool.process_high_level_action("Where is the mug?", {})
    assert env.belief_update_log[-1][0] == "AskHuman"
    assert "Asked human" in response
