import transformer_lens as lens
from typing import Dict, Tuple, List, Literal, Callable, Any
import torch
import random
from .graphed_model import GraphedModel, ComputeGraph
import torch.nn.functional as F

# TODO: figure out why ablations are fucked

all_tensors = {}

def autocircuit(
        model: GraphedModel,
        graph: ComputeGraph,
        root_n: str,
        prompts: torch.Tensor,
        cf_acts: Dict[str, torch.Tensor],
        threshold: float,
        ablation_mode: Literal["zero", "sample", "mean"] = "sample",
        diff_fn = None) -> ComputeGraph:
    # do a BFS on important nodes
    done = []
    queue = [root_n]
    ablated = []

    if diff_fn is None:
        diff_fn = F.cross_entropy

    def ablate_node(child):
        def go(tensor, hook=None):
            all_tensors[child] = tensor.clone()

            replacement = None
            if ablation_mode == "zero":
                replacement = torch.zeros_like(tensor)
            elif ablation_mode == "sample":
                replacement = cf_acts[child][random.randint(0, cf_acts[child].size(0) - 1)].repeat(tensor.size(0))
            elif ablation_mode == "mean":
                replacement = cf_acts[child].mean(0).repeat(tensor.size(0))
            return replacement
        return go

    while len(queue) > 0:
        node = queue.pop(0)

        baseline = model.run_with_hooks(prompts, ablated)

        if node in done:
            continue
        
        important = []

        sum_diff = 0
        n_children = 0

        for child in graph.get_children(node):
            print(f"looking at child {child} of {node}")

            to_ablate = ablated + [(graph.nodes[child], ablate_node(child))]

            out = model.run_with_hooks(prompts, to_ablate)
            print(out, baseline)
            diff = abs(diff_fn(out, baseline).item())

            sum_diff += diff
            n_children += 1

            if diff > threshold:
                important.append(child)
        
        if n_children > 0:
            print(node, "has important children", important, "and mean diff", sum_diff / n_children)
        queue.extend(important)
        done.append(node)
    
    return ComputeGraph.prune_nodes(graph, done)

# model needs to have run_with_hooks method
def path_tracing(
        model: GraphedModel,
        graph: ComputeGraph,
        root_n: str,
        prompts: torch.Tensor,
        normal_activations: Dict[str, torch.Tensor],
        counterfactual_activations: Dict[str, torch.Tensor],
        diff_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        threashold: float,
        ablation_mode: Literal["zero", "sample", "mean"],
        cache=None) -> ComputeGraph:
    # recursively find all nodes that affect the output of `root_n`

    if cache is None:
        cache = {}

    print("tracing", root_n)

    child_nodes = graph.get_children(root_n)
    if len(child_nodes) == 0:
        print(root_n, graph.nodes[root_n])
        graph = ComputeGraph({root_n: graph.nodes[root_n]}, [], root_n)
        cache[root_n] = graph
        return graph

    if root_n in cache:
        print("cache hit")
        return cache[root_n]

    important = []

    # TODO: probably compress this iteration into a single forward pass
    for child_n in child_nodes:
        child_hook = graph.nodes[child_n]
        child_cf = counterfactual_activations[child_n]
        #print(parent_cf)
        # rerun the network with the parent node ablated
        def ablate_node(hook_tensor):
            replacement = None
            if ablation_mode == "zero":
                replacement = torch.zeros_like(hook_tensor)
            elif ablation_mode == "sample":
                replacement = child_cf[random.randint(0, child_cf.size(0)-1)].repeat(hook_tensor.size(0))
            elif ablation_mode == "mean":
                replacement = child_cf.mean(dim=0).repeat(hook_tensor.size(0))
            
            if replacement is None:
                raise ValueError("Invalid ablation mode")
            
            return replacement

        ablated_activation = model.run_with_hooks(prompts, fwd_hooks=[(child_hook, ablate_node)], return_hook=root_n)
        normal_activation = normal_activations[root_n]

        diff = diff_fn(ablated_activation, normal_activation).item()
        if diff > threashold:
            print(f"diff {diff} > {threashold} for {child_n} of {root_n}")
            important.append(child_n)

    # recursively call path_tracing on the important nodes
    subgraphs = []
    for parent in important:
        subgraph = path_tracing(model, graph, parent, prompts, normal_activations, counterfactual_activations, diff_fn, threashold, ablation_mode, cache)
        subgraphs.append(subgraph)
        
    graph = ComputeGraph.merge_subgraphs(subgraphs, (root_n, graph.nodes[root_n]))
    cache[root_n] = graph
    return graph