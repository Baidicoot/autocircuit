import transformer_lens as lens
from typing import Dict, Tuple, List, Str, Literal, Callable
import torch
from typeguard import typechecked

# a ComputeGraph is a DAG that represents the causal relationship between hooks/nodes
# the path tracing algorithm outputs a simpler graph that is a subset of the input graph

class ComputeGraph:
    def __init__(self, nodes: Dict[str, lens.HookPoint], edges: Tuple[str, str], root: str):
        self.nodes = nodes
        self.edges = edges
        self.root = root
        self.no_cycles()
    
    def no_cycles(self):
        # cycle detection algorithm
        visited = set()
        stack = set()
        def dfs(node):
            visited.add(node)
            stack.add(node)
            for edge in self.edges:
                if edge[0] == node:
                    dfs(edge[1])
                elif edge[1] in stack:
                    raise ValueError(f"Cycle detected with node {edge[1]}")
            stack.remove(node)
        dfs(self.root)
    
    def get_parents(self, node):
        return [edge[0] for edge in self.edges if edge[1] == node]

@typechecked
def path_tracing(net: Callable[[List[Tuple[Callable, lens.HookPoint]]], None], graph: ComputeGraph, node: str, prompts: torch.Tensor, counterfactual_activations: Dict[str, torch.Tensor], threashold: float, ablation_mode: Literal["zero", "sample", "mean"]) -> ComputeGraph:
    # recursively find all nodes that affect the output of `node`
    parent_nodes = graph.get_parents(node)
    if len(parent_nodes) == 0:
        return ComputeGraph({node: graph.nodes[node]}, (), node)