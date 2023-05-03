import torch
from typing import Callable, Dict, Tuple, List, Any
from abc import ABC, abstractmethod

class Hook(ABC):
    @abstractmethod
    def add_hook(self, hook: Callable[[torch.Tensor], torch.Tensor]):
        pass

class HookManager(ABC):
    @property
    @abstractmethod
    def hooks(self) -> Dict[str, Hook]:
        pass

    @abstractmethod
    def remove_hooks(self):
        pass
    
    """
    @abstractmethod
    def get_layer(self, name: str) -> int:
        pass
    """

# a ComputeGraph is a DAG that represents the causal relationship between hooks/nodes
# the path tracing algorithm outputs a simpler graph that is a subset of the input graph
class ComputeGraph:
    def __init__(self, nodes: List[str], edges: Tuple[str, str], root: str):
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
    def prune_nodes(cls, graph, nodes_to_keep):
        nodes = [node for node in graph.nodes if node in nodes_to_keep]
        edges = [edge for edge in graph.edges if edge[0] in nodes_to_keep and edge[1] in nodes_to_keep]
        return cls(nodes, edges, graph.root)

class GraphedModel:
    def __init__(self, model, graph: ComputeGraph, hooks: HookManager):
        self.model = model
        self.hook_manager = hooks
        self.graph = graph
    
    def run_with_hooks(self, inputs: torch.Tensor, fwd_hooks: List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]], return_hook: str = None) -> torch.Tensor:
        self.hook_manager.remove_hooks()
        for node, hook in fwd_hooks:
            self.hook_manager.hooks[node].add_hook(hook)
        out = None
        if return_hook is not None:
            def output_hook(x):
                nonlocal out
                out = x
                return x
            self.hook_manager.hooks[return_hook].add_hook(output_hook)
            if hasattr(self.hook_manager, "get_layer"):
                self.model(inputs, stop_at_layer=self.hook_manager.get_layer(return_hook))
            else:
                self.model(inputs)
            self.hook_manager.remove_hooks()
            return out
        else:
            out = self.model(inputs)
            self.hook_manager.remove_hooks()
            return out
    
    def run_with_cache(self, inputs: torch.Tensor) -> torch.Tensor:
        fwd_hooks = []
        cache = {}
        for node in self.graph.nodes:
            def hook(x, n):
                nonlocal cache
                cache[n] = x
                return x
            fwd_hooks.append((node, lambda x, n=node : hook(x, n)))
        return self.run_with_hooks(inputs, fwd_hooks), cache