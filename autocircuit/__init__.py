import transformer_lens as lens
from typing import Dict, Tuple, List, Literal, Callable, Any
import torch
from typeguard import typechecked

# thing representing some (contiguous) selection of activations in a network
class PartialHook:
    def __init__(self, hook_point: str, hook_read: Callable[torch.Tensor, torch.Tensor], hook_repl: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.hook_point = hook_point
        self.hook_read = hook_read
        self.hook_repl = hook_repl

# a ComputeGraph is a DAG that represents the causal relationship between hooks/nodes
# the path tracing algorithm outputs a simpler graph that is a subset of the input graph
class ComputeGraph:
    def __init__(self, nodes: Dict[str, PartialHook], edges: Tuple[str, str], root: str):
        self.nodes = nodes
        self.edges = edges
        self.root = root

        self.no_cycles()
        self.all_reachable()
    
    def no_cycles(self):
        # cycle detection algorithm
        stack = []
        def dfs(node):
            stack.append(node)
            for edge in self.edges:
                if edge[0] == node:
                    if edge[1] in stack:
                        raise ValueError(f"Cycle detected with node {edge[1]}", stack)
                    dfs(edge[1])
            stack.pop()
        dfs(self.root)
    
    def all_reachable(self):
        # check all reachable from root
        visited = set()
        def dfs(node):
            visited.add(node)
            for edge in self.edges:
                if edge[0] == node:
                    dfs(edge[1])
        dfs(self.root)
        assert len(visited) == len(self.nodes), "Not all nodes are reachable from root"
    
    def get_parents(self, node):
        return [edge[1] for edge in self.edges if edge[0] == node]

    @classmethod
    def merge_many(cls, graphs, root):
        nodes = {root[0]: root[1]}
        edges = [(graph.root, root[0]) for graph in graphs]
        roots = [graph.root for graph in graphs]
        for graph in graphs:
            nodes.update(graph.nodes)
            edges.extend(graph.edges)
        return cls(nodes, edges, root[0])

# model needs to have run_with_hooks method
def path_tracing(model, graph: ComputeGraph, node: str, prompts: torch.Tensor, counterfactual_activations: Dict[str, torch.Tensor], diff_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], threshold: float, ablation_mode: Literal["zero", "sample", "mean"]) -> ComputeGraph:
    # recursively find all nodes that affect the output of `node`
    parent_nodes = graph.get_parents(node)
    if len(parent_nodes) == 0:
        return ComputeGraph({node: graph.nodes[node]}, (), node)
    
    important = []

    # TODO: probably compress this iteration into a single forward pass
    for parent_n in parent_nodes:
        parent = graph.nodes[parent_n]
        parent_cf = parent.hook_read(counterfactual_activations[graph.nodes[parent].hook_point])
        # rerun the network with the parent node ablated
        def ablate_node(hook_tensor, hook_point):
            replacement = None
            if ablation_mode == "zero":
                replacement = torch.zeros_like(parent_cf)
            elif ablation_mode == "sample":
                replacement = parent_cf[random.randint(0, parent_cf.size(0))].repeat(parent_cf.size(0))
            elif ablation_mode == "mean":
                replacement = parent_cf.mean(dim=0).repeat(parent_cf.size(0))
            
            if replacement is None:
                raise ValueError("Invalid ablation mode")
            
            return parent.hook_repl(hook_tensor, replacement)

        diff = diff_fn(model.run_with_hooks(prompts, fwd_hooks=[(parent.hook_point, ablate_node)]), model(prompts)).item()
        print(f"ablated {parent_n}")
        if loss > threshold:
            important.append(parent)
    
    # recursively call path_tracing on the important nodes
    subgraphs = []
    for parent in important:
        subgraph = path_tracing(net, graph, parent, prompts, counterfactual_activations, loss_fn, threashold, ablation_mode)
        subgraphs.append(subgraph)
    
    return ComputeGraph.merge_many(subgraphs, root)