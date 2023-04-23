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
        self.all_reachable()
    
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
    
    def all_reachable(self):
        # all reachable from root
        visited = set()
        def dfs(node):
            visited.add(node)
            for edge in self.edges:
                if edge[0] == node:
                    dfs(edge[1])
        dfs(self.root)
        assert len(visited) == len(self.nodes), "Not all nodes are reachable from root"
    
    def get_parents(self, node):
        return [edge[0] for edge in self.edges if edge[1] == node]

    @classmethod
    def merge_many(cls, graphs: List[ComputeGraph], root: str):
        nodes = {}
        edges = []
        for graph in graphs:
            nodes.update(graph.nodes)
            edges.extend(graph.edges)
        return cls(nodes, edges, root)

def path_tracing(net: Callable[[List[Tuple[Callable, lens.HookPoint]], torch.Tensor], torch.Tensor], graph: ComputeGraph, node: str, prompts: torch.Tensor, counterfactual_activations: Dict[str, torch.Tensor], loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], threshold: float, ablation_mode: Literal["zero", "sample", "mean"]) -> ComputeGraph:
    # recursively find all nodes that affect the output of `node`
    parent_nodes = graph.get_parents(node)
    if len(parent_nodes) == 0:
        return ComputeGraph({node: graph.nodes[node]}, (), node)
    
    important = []

    # TODO: probably compress this iteration into a single forward pass
    for parent in parent_nodes:
        # rerun the network with the parent node ablated
        def ablate_node(hook_tensor, hook_point):
            if ablation_mode == "zero":
                return torch.zeros_like(hook_tensor)
            elif ablation_mode == "sample":
                return counterfactual_activations[parent][random.randint(0, counterfactual_activations.size(0))].repeat(hook_tensor.size(0))
            elif ablation_mode == "mean":
                return counterfactual_activations[parent].mean(dim=0).repeat(hook_tensor.size(0))
        
        loss = loss_fn(net([(ablate_node, graph.nodes[parent])], prompts), net([], prompts)).item()
        if loss > threshold:
            important.append(parent)
    
    # recursively call path_tracing on the important nodes
    subgraphs = []
    for parent in important:
        subgraph = path_tracing(net, graph, parent, prompts, counterfactual_activations, loss_fn, threashold, ablation_mode)
        subgraphs.append(subgraph)
    
    return ComputeGraph.merge_many(subgraphs, root)