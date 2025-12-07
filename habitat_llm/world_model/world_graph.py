#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np

from habitat_llm.world_model.entity import (
    Concept,
    Entity,
    House,
    Human,
    Object,
    Receptacle,
    Room,
    SpotRobot,
)
from habitat_llm.world_model.entities.furniture import Furniture
from habitat_llm.world_model.graph import Graph


def flip_edge(edge: str) -> str:
    return {
        "next to": "next to",
        "on": "under",
        "in": "has",
        "inside": "contains",
        "describes": "described_by",
        "represents": "represented_by",
    }.get(edge, "unknown")


class WorldGraph(Graph):
    """
    This class represents robot's model of the world.
    This could be populated from the robot's perception stack
    or ground truth simulator info. As implemented here,
    WorldModel is a Directed Acyclic Graph.
    """

    # Parameterized Constructor
    def __init__(self, graph=None, graph_type: Optional[str] = None):
        # Create a graph to store different entities in the world
        # and their relations to one another
        super().__init__(graph=graph)
        self.agent_asymmetry = False
        self.world_model_type = "privileged"
        # Track which agent's belief the graph represents (e.g., robot/human).
        # This attribute is read during deepcopy, so always initialize it even
        # when not explicitly provided.
        self.graph_type: str = graph_type or "robot"
        self.concept_confidence: dict = {}
        self.belief_divergence: float = 0.0
        self._logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self._logger.setLevel(logging.DEBUG)

    def get_all_rooms(self):
        """
        This method returns all rooms in the world graph
        """
        return [node for node in self.graph if isinstance(node, Room)]

    def get_all_receptacles(self):
        """
        This method returns all receptacles in the world graph
        """
        return [node for node in self.graph if isinstance(node, Receptacle)]

    def get_all_furnitures(self):
        """
        This method returns all surfaces in the world graph
        """
        return [node for node in self.graph if isinstance(node, Furniture)]

    def get_all_objects(self):
        """
        This method returns all objects in the world graph
        """
        return [node for node in self.graph if isinstance(node, Object)]

    def add_or_update_concept_annotation(
        self,
        target: Union[str, Entity],
        concept_labels: List[str],
        concept_confidence: List[float],
        concept_node_name: Optional[str] = None,
        relation: str = "describes",
    ) -> Concept:
        """Attach concept annotations (labels + confidences) to a node and optional concept node.

        Args:
            target: The entity or entity name to annotate.
            concept_labels: Concept labels describing the target.
            concept_confidence: Confidence values aligned with ``concept_labels``.
            concept_node_name: Optional explicit name for the concept node. Defaults to
                ``concept_<target>`` when omitted.
            relation: Edge label used between concept and target nodes.
        """

        if len(concept_labels) != len(concept_confidence):
            raise ValueError("concept_labels and concept_confidence must be the same length")

        if isinstance(target, str):
            target_node = self.get_node_from_name(target)
        else:
            target_node = target

        merged_labels, merged_confidence = self._merge_concept_properties(
            target_node.properties.get("concept_labels", []),
            target_node.properties.get("concept_confidence", []),
            concept_labels,
            concept_confidence,
        )
        target_node.properties["concept_labels"] = merged_labels
        target_node.properties["concept_confidence"] = merged_confidence

        concept_name = concept_node_name or f"concept_{target_node.name}"
        try:
            concept_node = self.get_node_from_name(concept_name)
            if not isinstance(concept_node, Concept):
                raise ValueError(f"Existing node {concept_name} is not a Concept")
        except ValueError:
            concept_node = Concept(
                concept_name,
                {
                    "type": "concept",
                    "concept_labels": merged_labels,
                    "concept_confidence": merged_confidence,
                },
            )
            self.add_node(concept_node)
        else:
            concept_node.properties["concept_labels"] = merged_labels
            concept_node.properties["concept_confidence"] = merged_confidence

        opposite_relation = flip_edge(relation)
        if opposite_relation == "unknown":
            opposite_relation = relation
        self.add_edge(concept_node, target_node, relation, opposite_relation)
        return concept_node

    @staticmethod
    def _merge_concept_properties(
        existing_labels: List[str],
        existing_confidence: List[float],
        new_labels: List[str],
        new_confidence: List[float],
    ) -> tuple[list[str], list[float]]:
        """Merge concept labels/confidences preferring highest confidence per label."""

        confidence_map = {label: conf for label, conf in zip(existing_labels, existing_confidence)}
        for label, conf in zip(new_labels, new_confidence):
            if label in confidence_map:
                confidence_map[label] = max(confidence_map[label], conf)
            else:
                confidence_map[label] = conf
        merged_labels = list(confidence_map.keys())
        merged_confidence = [confidence_map[label] for label in merged_labels]
        return merged_labels, merged_confidence

    def get_node_with_property(self, property_key, property_val):
        """
        This method returns a node in the world graph that
        matches given type and having given property
        """
        for node in self.graph:
            if (property_key in node.properties) and (
                node.properties[property_key] == property_val
            ):
                return node

        self._logger.info(
            f"World graph does not have a node having property {property_key} with {property_val}"
        )

        return None

    def get_spot_robot(self):
        """
        This method returns spot robot node
        """
        for node in self.graph:
            if isinstance(node, SpotRobot):
                return node

        raise ValueError("World graph does not contain a node of type SpotRobot")

    def get_human(self):
        """
        This method returns human node
        """
        for node in self.graph:
            if isinstance(node, Human):
                return node

        raise ValueError("World graph does not contain a node of type Human")

    def get_agents(self):
        """
        This method returns all agent nodes
        """
        out = []
        for node in self.graph:
            if isinstance(node, (Human, SpotRobot)):
                out.append(node)

        if len(out) == 0:
            raise ValueError(
                "World graph does not contain a node of type Human or SpotRobot"
            )

        return out

    def get_room_for_entity(self, entity):
        """
        This method returns the room in which the given entity is
        """

        # Get nodes of type room
        room = self.get_neighbors_of_type(entity, Room)

        if room is None or len(room) == 0:
            raise ValueError(f"No room found for entity {entity}")

        if len(room) > 1:
            self._logger.info(
                f"Multiple rooms found for entity {entity}, returning only one room"
            )
            return room[0]

        return room[0]

    def get_closest_object_or_furniture(
        self, obj_node, n: int, dist_threshold: float = 1.5
    ) -> List[Union[Object, Furniture]]:
        """
        This method returns n closest objects or furnitures to the given object node
        """
        closest = sorted(
            self.get_all_objects() + self.get_all_furnitures(),
            key=lambda x: np.linalg.norm(
                np.array(obj_node.properties["translation"])
                - np.array(x.properties["translation"])
            ),
        )[:n]
        within_threshold = [
            obj
            for obj in closest
            if np.linalg.norm(
                np.array(obj_node.properties["translation"])
                - np.array(obj.properties["translation"])
            )
            < dist_threshold
        ]
        return within_threshold

    def average_concept_confidence(self) -> float:
        """Return the average concept confidence tracked for this graph.

        Defaults to 1.0 when no confidences have been recorded so existing
        callers can treat the metric as fully confident in the absence of
        explicit values.
        """

        if not self.concept_confidence:
            return 1.0
        return float(sum(self.concept_confidence.values()) / len(self.concept_confidence))

    def compute_belief_divergence(self, other_graph: "WorldGraph") -> float:
        """Estimate divergence between this graph and another agent's graph.

        The metric compares object-to-furniture assignments across graphs and
        reports the fraction of mismatched pairs. A value of 0.0 indicates
        alignment while larger values highlight disagreement on object
        placement.
        """

        # Build lookup for object locations in self and other
        def _object_location_pairs(graph: "WorldGraph"):
            pairs = {}
            for obj, furn in graph.get_objects_and_their_furnitures().items():
                pairs[obj.name] = furn.name
            return pairs

        reference_pairs = _object_location_pairs(self)
        other_pairs = _object_location_pairs(other_graph)
        if not reference_pairs and not other_pairs:
            return 0.0

        object_union = set(reference_pairs) | set(other_pairs)
        if not object_union:
            return 0.0

        mismatches = 0
        for obj_name in object_union:
            if reference_pairs.get(obj_name) != other_pairs.get(obj_name):
                mismatches += 1

        divergence = mismatches / len(object_union)
        self.belief_divergence = divergence
        return divergence

    # TODO: [BE] This function is duplicated in instruct/utils.py. Should be refactored
    # to avoid duplication and maintainability issues.
    def get_world_descr(self, is_human_wg: bool = False):
        ## house description -- rooms and their furniture list
        furn_room = self.group_furniture_by_room()
        house_info = ""
        for k, v in furn_room.items():
            furn_names = [furn.name for furn in v]
            all_furn = ", ".join(furn_names)
            house_info += k + ": " + all_furn + "\n"

        ## get objects held by the agent
        spot_node = self.get_spot_robot()
        human_node = self.get_human()

        ## locations of objects in the house
        objs_info = ""
        all_objs = self.get_all_objects()
        for obj in all_objs:
            if self.is_object_with_agent(obj, agent_type="robot"):
                objs_info += obj.name + ": " + spot_node.name + "\n"
            elif self.is_object_with_agent(obj, agent_type="human"):
                objs_info += obj.name + ": " + human_node.name + "\n"
            else:
                furniture = self.find_furniture_for_object(obj)
                if furniture is not None:
                    objs_info += obj.name + ": " + furniture.name + "\n"
                elif furniture is None and (
                    (is_human_wg and self.agent_asymmetry)
                    or (not is_human_wg and self.world_model_type == "concept_graph")
                ):
                    # Objects are allowed to be marooned on unknown furniture under
                    # agent asymmetry condition, since the object may be placed anywhere
                    # in the house unbeknownst to the human agent
                    objs_info += obj.name + ": " + "unknown" + "\n"
                else:
                    raise ValueError(f"Object {obj.name} has no parent")
        return f"Furniture:\n{house_info}\nObjects:\n{objs_info}"

    def is_object_with_human(self, obj):
        """
        This method checks if the object is connected to any agent
        """
        # Fetch node if input type is string
        if isinstance(obj, str):
            obj = self.get_node_from_name(obj)

        return any(isinstance(neighbor, (Human)) for neighbor in self.graph[obj])

    def is_object_with_robot(self, obj):
        """
        This method checks if the object is connected to any agent
        """
        # Fetch node if input type is string
        if isinstance(obj, str):
            obj = self.get_node_from_name(obj)

        return any(isinstance(neighbor, (SpotRobot)) for neighbor in self.graph[obj])

    def is_object_with_agent(self, obj, agent_type="any"):
        """
        This method checks if the object is connected to any agent
        """
        # Fetch node if input type is string
        if isinstance(obj, str):
            obj = self.get_node_from_name(obj)
        return_dict = {
            "any": any(
                isinstance(neighbor, (SpotRobot, Human)) for neighbor in self.graph[obj]
            ),
            "human": any(isinstance(neighbor, (Human)) for neighbor in self.graph[obj]),
            "robot": any(
                isinstance(neighbor, (SpotRobot)) for neighbor in self.graph[obj]
            ),
        }
        if agent_type in return_dict:
            return return_dict[agent_type]
        else:
            raise ValueError(f"Agent type {agent_type} not recognized.")

    def find_object_furniture_pairs(self):
        """
        This method returns dictionary of all objects
        and their parent furniture or rooms
        """
        pairs = {}
        for node, neighbors in self.graph.items():
            if isinstance(node, Object):
                for neighbor in neighbors:
                    if isinstance(neighbor, Receptacle):
                        for second_neighbor in self.graph[neighbor]:
                            if isinstance(second_neighbor, Furniture):
                                pairs[node] = second_neighbor
                    elif isinstance(neighbor, Furniture):
                        pairs[node] = neighbor

        return pairs

    def find_furniture_for_object(self, obj: Object, verbose: bool = False):
        """
        This method returns Furniture associated with the given object
        """
        for neighbor in self.graph[obj]:
            if isinstance(neighbor, Receptacle):
                for second_neighbor in self.graph[neighbor]:
                    if isinstance(second_neighbor, Furniture):
                        return second_neighbor
            elif isinstance(neighbor, Furniture):
                return neighbor

        if verbose:
            self._logger.info(
                f"No furniture for object with name {obj.name} was found in the graph"
            )
        return None

    def find_receptacle_for_object(self, obj):
        """
        Get the Receptacle Entity for an Object.
        """
        for neighbor in self.graph[obj]:
            if isinstance(neighbor, Receptacle):
                return neighbor
        return None

    def find_furniture_for_receptacle(self, rec):
        """
        This method returns Furniture associated with the given receptacle
        """
        for neighbor in self.graph[rec]:
            if isinstance(neighbor, Furniture):
                return neighbor

        raise ValueError(
            f"No furniture for receptacle with name {rec.name} was found in the graph"
        )

    def group_furniture_by_type(self):
        """
        Groups Furniture nodes by their types
        """
        furniture_by_type = {}
        for node in self.graph:
            if isinstance(node, Furniture):
                fur_type = node.properties["type"]
                if fur_type in furniture_by_type:
                    furniture_by_type[fur_type].append(node)
                else:
                    furniture_by_type[fur_type] = [node]
        return furniture_by_type

    def group_furniture_by_room(self):
        """
        Groups Furniture nodes by their rooms
        """
        furniture_by_room = defaultdict(list)
        for node in self.graph:
            if isinstance(node, Furniture):
                for neighbor in self.graph[node]:
                    if isinstance(neighbor, Room):
                        furniture_by_room[neighbor.name].append(node)

        return furniture_by_room

    def group_furniture_by_room_type(self):
        """
        Groups Furniture nodes by their room types
        """
        furniture_by_room = {}
        for node in self.graph:
            if isinstance(node, Furniture):
                for neighbor in self.graph[node]:
                    if isinstance(neighbor, Room):
                        if neighbor.properties["type"] in furniture_by_room:
                            furniture_by_room[neighbor.properties["type"]].append(node)
                        else:
                            furniture_by_room[neighbor.properties["type"]] = [node]

        return furniture_by_room

    def get_furniture_to_room_map(self):
        """
        Returns dictionary of furniture node to room nodes
        """
        furniture_to_room = {}
        for node in self.graph:
            if isinstance(node, Furniture):
                for neighbor in self.graph[node]:
                    if isinstance(neighbor, Room):
                        furniture_to_room[node] = neighbor
                        break

        return furniture_to_room

    def get_furniture_in_room(self, room_node):
        """
        Returns list of all furniture nodes in a given room
        """
        if isinstance(room_node, str):
            room_node = self.get_node_from_name(room_node)

        # Get all neighbors of the room node with type Furniture
        furniture_list = self.get_neighbors_of_type(room_node, Furniture)

        return furniture_list

    def update(self, recent_graph, partial_obs, update_mode, add_only: bool = False):
        """
        This method updates the graph based on the recent_graph.
        recent_graph contains either the entire or subgraphs of
        the ground truth graph.
        Currently, this method is performing handle based data association.


        NOTE: In future, we should probably do position based data association,
        as the handles may be arbitrary when coming from actual perception pipeline.
        """
        # Throw if not operating in ground truth mode
        if update_mode != "gt":
            raise ValueError(
                f"World Graph can currently only be updated in ground truth mode, received: {update_mode}"
            )

        # Replace graph with the updated one
        # if operating in full observability
        if not partial_obs:
            self.graph = self.deepcopy_graph(recent_graph.graph)
        else:
            # if operating in partial observability
            self.merge(recent_graph, add_only=add_only)

        # update agent's properties if it is holding an object
        # episode may be single-agent with robot-only; handle that
        human_node = self.get_all_nodes_of_type(Human)
        human_object_nodes = []
        if human_node:
            human_node = human_node[0]
            human_object_nodes = self.get_neighbors_of_type(human_node, Object)
        else:
            human_node = None

        # episode may be single-agent with human-only; handle that
        robot_node = self.get_all_nodes_of_type(SpotRobot)
        robot_object_nodes = []
        if robot_node:
            robot_node = robot_node[0]
            robot_object_nodes = self.get_neighbors_of_type(robot_node, Object)
        else:
            robot_node = None

        if len(human_object_nodes) > 0:
            human_node.properties["last_held_object"] = human_object_nodes[0]
        if len(robot_object_nodes) > 0:
            robot_node.properties["last_held_object"] = robot_object_nodes[0]

        return

    def find_path(
        self,
        root_node: Union[str, Entity] = "house",
        end_node_types: list = None,
        visited: set = None,
        verbose: bool = False,
    ) -> Optional[dict]:
        """
        This method returns the path from the given node to the first node of type
        in end_node_types. It uses DFS to find the path.
        """
        if end_node_types is None:
            end_node_types = [Room]
        if isinstance(root_node, str):
            root_node = self.get_node_from_name(root_node)
            if verbose:
                self._logger.info(
                    f"Finding path from {root_node.name} to {end_node_types=}"
                )

        if isinstance(root_node, tuple(end_node_types)):
            return {}  # Return empty path if we are already at the end node

        if visited is None:
            visited = set()

        for neighbor, edge in self.graph[root_node].items():
            if neighbor not in visited:
                visited.add(neighbor)
                path = self.find_path(neighbor, end_node_types, visited)
                if path is not None:
                    if root_node in path:
                        path[root_node][neighbor] = edge
                    else:
                        path[root_node] = {neighbor: edge}
                    if neighbor in path:
                        path[neighbor][root_node] = self.graph[neighbor][root_node]
                    else:
                        path[neighbor] = {root_node: self.graph[neighbor][root_node]}
                    return path
        return None

    def get_subgraph(self, nodes_in, verbose: bool = False):
        """
        Method to get subgraph over objects in the view and agents.
        The relevant subgraph is considered the path from object to closest furniture,
        from agent to object-in-hand and from agent to the room they are in.

        Input is a list of name of entities in the agent's view. We sort through them and
        only keep objects. We then find a path from each object to the first Furniture node,
        which is called that object's relevant-subgraph. This relevant subgraph is then
        used to add/update objects in the world graph.
        """

        # Initialize empty subgraph
        subgraph = Graph()

        # Create root node
        house = House("house", {"type": "root"}, "house_0")
        subgraph.add_node(house)

        # Create list of nodes if input is list of strings
        nodes = []
        for node in nodes_in:
            curr_node = self.get_node_from_name(node) if isinstance(node, str) else node
            if isinstance(curr_node, (Object, Human, SpotRobot)):
                if verbose:
                    self._logger.info(
                        f"Adding {curr_node.name}, {curr_node.properties['type']} to recent subgraph"
                    )
                nodes.append(curr_node)

        # add all required nodes in the subgraph
        for curr_node in nodes:
            subgraph.add_node(curr_node)

        # Loop through all object+agent nodes
        # and populate edges in the subgraph up to House
        for curr_node in nodes:
            path_graph = self.find_path(
                root_node=curr_node,
                end_node_types=[House],
                verbose=True,
            )

            if path_graph is not None:
                for curr_node in path_graph:
                    subgraph.add_node(curr_node)
                    for neighbor, edge in path_graph[curr_node].items():
                        if neighbor not in nodes:
                            subgraph.add_node(neighbor)
                        subgraph.add_edge(
                            curr_node, neighbor, edge, path_graph[neighbor][curr_node]
                        )
                        if verbose:
                            self._logger.info(
                                f"Added edge: {curr_node.name} {neighbor.name} {edge} {path_graph[neighbor][curr_node]}"
                            )
            else:
                if verbose:
                    self._logger.info(
                        f"No path found {curr_node.name=}, {type(curr_node)}!!!"
                    )

        return subgraph

    def to_dot(self):
        """
        Helpful utility to convert graph to dot format for visualization

        Note: Couldn't find a great way to visualize the output yet though, seems very chaotic
        """
        dot = "digraph {\n"
        for node in self.graph:
            for neighbor, edge in self.graph[node].items():
                dot += f'    "{node}" -> "{neighbor}" [label="{edge}"];\n'
        dot += "}"
        return dot

    def serialize_concept_layer(self) -> dict:
        """Serialize concept annotations for downstream logging or planner use."""

        entity_concepts = []
        concept_nodes = []
        for node in self.graph:
            labels = node.properties.get("concept_labels")
            confidence = node.properties.get("concept_confidence")
            if labels is not None and confidence is not None:
                record = {
                    "name": node.name,
                    "type": node.properties.get("type"),
                    "concept_labels": labels,
                    "concept_confidence": confidence,
                }
                if isinstance(node, Concept):
                    concept_nodes.append(record)
                else:
                    entity_concepts.append(record)

        return {
            "entity_concepts": entity_concepts,
            "concept_nodes": concept_nodes,
        }

    def log_concept_layer(self, log_path: Optional[str] = None) -> dict:
        """Serialize and optionally write concept layer to disk for debugging."""

        serialized = self.serialize_concept_layer()
        serialized_str = json.dumps(serialized, indent=2, default=str)
        self._logger.debug("Concept layer: %s", serialized_str)
        if log_path is not None:
            with open(log_path, "w") as f:
                f.write(serialized_str)
        return serialized

    def __deepcopy__(self, memo):
        """
        Method to deep copy this instance
        """
        graph_copy = super().__deepcopy__(memo)
        wg = WorldGraph(graph_type=self.graph_type)
        wg.graph = graph_copy.graph
        return wg

    def get_closest_entities(
        self,
        n: int,
        object_node: Entity = None,
        location: list = None,
        dist_threshold: float = 1.5,
        include_rooms: bool = False,
        include_furniture: bool = True,
        include_objects: bool = True,
    ) -> List[Union[Object, Furniture, Room]]:
        """
        This method returns n closest objects or furnitures to the given object node, or
        given location, within a distance threshold from the given input.
        If dist_threshold is negative or zero, it returns n closest entities regardless
        of distance.
        """
        # TODO: add an optional arg include_rooms:bool and also return rooms in this list
        if object_node is None and location is None:
            raise ValueError("Either object_node or location should be provided")
        if location is not None and object_node is not None:
            self._logger.debug(
                "Provided both object_node and location. Only object-node information will be used to get closest entities."
            )
        if object_node is not None:
            location = np.array(object_node.properties["translation"])
        elif location is not None:
            if len(location) != 3:
                raise ValueError("Location should be a list of 3 elements")
            location = np.array(location)

        entity_list = []
        if include_rooms:
            entity_list += self.get_all_rooms()
        if include_furniture:
            entity_list += self.get_all_furnitures()
        if include_objects:
            entity_list += self.get_all_objects()
        filtered_entity_list = []
        for ent in entity_list:
            if "translation" in ent.properties:
                filtered_entity_list.append(ent)
            else:
                self._logger.debug(
                    f"Entity {ent.name} found without a translation property"
                )
        entity_list = filtered_entity_list
        closest = sorted(
            entity_list,
            key=lambda x: np.linalg.norm(
                location - np.array(x.properties["translation"])
            ),
        )[:n]
        if dist_threshold > 0.0:
            within_threshold = [
                obj
                for obj in closest
                if np.linalg.norm(location - np.array(obj.properties["translation"]))
                < dist_threshold
            ]
        else:
            within_threshold = closest
        return within_threshold


class BeliefGraphContainer:
    """Container that keeps robot and human belief graphs in sync.

    It provides helper APIs to update each belief independently, switch the active
    view, and compute divergence between the two graphs.
    """

    def __init__(
        self,
        robot_graph: Optional[WorldGraph] = None,
        human_graph: Optional[WorldGraph] = None,
    ) -> None:
        self._graphs: Dict[str, WorldGraph] = {
            "robot": robot_graph or WorldGraph(graph_type="robot"),
            "human": human_graph or WorldGraph(graph_type="human"),
        }
        # Track the active graph type for convenience (defaults to robot)
        self.active_graph_type: str = "robot"

    def set_active_graph(self, graph_type: str) -> None:
        if graph_type not in self._graphs:
            raise ValueError(f"Unknown graph_type {graph_type}")
        self.active_graph_type = graph_type

    def get_graph(self, graph_type: Optional[str] = None) -> WorldGraph:
        graph_type = graph_type or self.active_graph_type
        if graph_type not in self._graphs:
            raise ValueError(f"Unknown graph_type {graph_type}")
        return self._graphs[graph_type]

    def update_robot_belief(
        self, recent_graph: Graph, partial_obs: bool, update_mode: str, add_only: bool = False
    ) -> None:
        self._graphs["robot"].update(recent_graph, partial_obs, update_mode, add_only)

    def update_human_belief(
        self, recent_graph: Graph, partial_obs: bool, update_mode: str, add_only: bool = False
    ) -> None:
        self._graphs["human"].update(recent_graph, partial_obs, update_mode, add_only)

    def sync_graphs(self, from_graph: str = "robot", to_graph: str = "human") -> None:
        """Copy belief state from one graph to the other."""

        if from_graph not in self._graphs or to_graph not in self._graphs:
            raise ValueError("sync_graphs requires valid graph keys: 'robot' or 'human'")
        self._graphs[to_graph] = deepcopy(self._graphs[from_graph])
        self._graphs[to_graph].graph_type = to_graph

    def compute_belief_divergence(self) -> Dict[str, float]:
        from habitat_llm.world_model.belief_divergence import compute_belief_divergence

        return compute_belief_divergence(
            self._graphs["robot"],
            self._graphs["human"],
        )
