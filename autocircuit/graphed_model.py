import torch
from typing import Callable, Dict, Tuple, List, Any
from abc import ABC, abstractmethod

class Hook(ABC):
    pass

# thing representing some (contiguous) selection of activations in a network
class PartialHook:
    def __init__(self, hook_point: str, hook_read: Callable[[torch.Tensor], torch.Tensor], hook_repl: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.hook_point = hook_point
        self.hook_read = hook_read
        # hook_repl is a function that takes a tensor and a replacement tensor and returns a tensor
        self.hook_repl = hook_repl

    @classmethod
    def process_hooks(cls, hooks: List[Tuple["PartialHook", Callable[[torch.Tensor], torch.Tensor]]]) -> List[Tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]]:
        by_hook_point = {}
        for hook, fn in hooks:
            if hook.hook_point not in by_hook_point:
                by_hook_point[hook.hook_point] = []
            by_hook_point[hook.hook_point].append((hook, fn))
        return [(hook_point, cls._merge_fns(fns)) for hook_point, fns in by_hook_point.items()]

    @classmethod
    def _merge_fns(cls, fns: List[Tuple["PartialHook", Callable[[torch.Tensor], torch.Tensor]]]) -> Callable[[torch.Tensor, Any], torch.Tensor]:
        def go(tensor, hook=None):
            before_shape = tensor.shape
            for hook, fn in fns:
                tensor = hook.hook_repl(tensor, fn(hook.hook_read(tensor)))
            assert tensor.shape == before_shape, f"Shape changed from {before_shape} to {tensor.shape} in {hook.hook_point}"
            return tensor
        return go

# a ComputeGraph is a DAG that represents the causal relationship between hooks/nodes
# the path tracing algorithm outputs a simpler graph that is a subset of the input graph
class ComputeGraph:
    def __init__(self, nodes: Dict[str, Hook], edges: Tuple[str, str], root: str):
        self.nodes = nodes
        self.edges = edges
        self.root = root

        #self.no_cycles()
        #self.all_reachable()
    
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
    
    def get_children(self, node):
        return [edge[1] for edge in self.edges if edge[0] == node]

    @classmethod
    def merge_subgraphs(cls, graphs, shared_root):
        nodes = {n: v for graph in graphs for n, v in graph.nodes.items()}
        edges = list({edge for graph in graphs for edge in graph.edges})
        nodes[shared_root[0]] = shared_root[1]
        for graph in graphs:
            edges.append((shared_root[0], graph.root))
        return cls(nodes, edges, shared_root[0])

    @classmethod
    def merge_many(cls, graphs, new_root):
        nodes = {n: v for graph in graphs for n, v in graph.nodes.items()}
        edges = list({edge for graph in graphs for edge in graph.edges})
        return cls(nodes, edges, new_root)

    @classmethod
    def prune_nodes(cls, graph, nodes_to_keep):
        nodes = {n: v for n, v in graph.nodes.items() if n in nodes_to_keep}
        edges = [edge for edge in graph.edges if edge[0] in nodes_to_keep and edge[1] in nodes_to_keep]
        return cls(nodes, edges, graph.root)

# model that corresponds to some compute graph
class GraphedModel(ABC):
    @abstractmethod
    def run_with_hooks(self, inputs: torch.Tensor, fwd_hooks: List[Tuple[str, Callable[[torch.Tensor, PartialHook], torch.Tensor]]], return_hook: str = None) -> torch.Tensor:
        raise NotImplementedError("run_with_hooks not implemented")

    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("__call__ not implemented")
    
    # return the FULL graph
    @property
    @abstractmethod
    def graph(self) -> ComputeGraph:
        raise NotImplementedError("graph not implemented")