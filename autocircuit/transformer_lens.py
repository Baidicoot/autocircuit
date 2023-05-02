# autocircuit for TransformerLens

from typing import Callable, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_lens as lens
import einops

from autocircuit.graphed_model import ComputeGraph, PartialHook

from .graphed_model import GraphedModel, ComputeGraph, PartialHook

def idx_token(tensor, t):
    return tensor[:, t]

# probably fine to do this in-place?
def replace_token(tensor, t, replacement):
    tensor[:, t] = replacement
    return tensor

def replace_head_token(tensor, t, h, replacement):
    tensor[:, t, h] = replacement
    return tensor

class Transformer(GraphedModel):
    def __init__(self, transformer, cfg, n_ctx):
        super().__init__()
        self.model = transformer
        self.n_ctx = n_ctx
        self.cfg = cfg
        self._graph, self.node_layers = Transformer.build_graph(cfg, n_ctx)

    @property
    def graph(self) -> ComputeGraph:
        return self._graph

    def run_with_hooks(self, inputs: torch.Tensor, fwd_hooks: List[Tuple[PartialHook, Callable[[torch.Tensor], torch.Tensor]]], return_hook: str = None) -> torch.Tensor:
        # run with hooks
        fwd_hooks = PartialHook.process_hooks(fwd_hooks)
        hooked_output = None
        if return_hook is not None:
            def go(x, hook=None):
                nonlocal hooked_output
                hooked_output = self.graph.nodes[return_hook].hook_read(x)
            fwd_hooks.append((self.graph.nodes[return_hook].hook_point, go))
            self.model.run_with_hooks(inputs, fwd_hooks=fwd_hooks, stop_at_layer=self.node_layers[return_hook]+1)
            return hooked_output
        else:
            return self.model.run_with_hooks(inputs, fwd_hooks=fwd_hooks)

    def __call__(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def run_with_cache(self, model, inputs):
        _, cache = model.run_with_cache(inputs)
        print(cache.keys())
        acts = {}
        for node in self.graph.nodes:
            print(node, cache[self.graph.nodes[node].hook_point].shape)
            acts[node] = self.graph.nodes[node].hook_read(cache[self.graph.nodes[node].hook_point])
            print(node, acts[node].shape)
        return acts

    # will need to cache important parents
    @staticmethod
    def build_graph(cfg, n_ctx) -> Tuple[ComputeGraph, Dict[str, int]]:
        nodes = {}
        edges = []
        node_layers = {}
        for t in range(n_ctx):
            for l in range(cfg.n_layers):
                # hook_resid_out
                nodes[f"{t}.{l}.resid_post"] = PartialHook(
                    f"blocks.{l}.hook_resid_post",
                    lambda x: idx_token(x, t),
                    lambda x, y: replace_token(x, t, y)
                )
                node_layers[f"{t}.{l}.resid_post"] = l
                # hook_resid_mid
                nodes[f"{t}.{l}.resid_mid"] = PartialHook(
                    f"blocks.{l}.hook_resid_mid",
                    lambda x: idx_token(x, t),
                    lambda x, y: replace_token(x, t, y)
                )
                node_layers[f"{t}.{l}.resid_mid"] = l
                # hook_mlp_out
                nodes[f"{t}.{l}.mlp_out"] = PartialHook(
                    f"blocks.{l}.hook_mlp_out",
                    lambda x: idx_token(x, t),
                    lambda x, y: replace_token(x, t, y)
                )
                node_layers[f"{t}.{l}.mlp_out"] = l
                # reader heads
                for h in range(cfg.n_heads):
                    # reader
                    nodes[f"{t}.{l}.reader.{h}"] = PartialHook(
                        f"blocks.{l}.hook_attn_out",
                        lambda x: x[:, t, cfg.d_head * h : cfg.d_head * (h + 1)],
                        lambda x, y:
                            einops.rearrange(
                                replace_head_token(einops.rearrange(
                                    x,
                                    "b p (i d) -> b p i d",
                                    d=cfg.d_head
                                ), t, h, y),
                                "b p i d -> b p (i d)"
                            )
                    )
                    node_layers[f"{t}.{l}.reader.{h}"] = l
                    """
                    # writer - q
                    nodes[f"{t}.{l}.writer.{h}.q"] = PartialHook(
                        f"blocks.{l}.attn.hook_q",
                        lambda x: x[:, t, h],
                        lambda x, y: replace_head_token(x, t, h, y)
                    )
                    node_layers[f"{t}.{l}.writer.{h}.q"] = l
                    # writer - k
                    nodes[f"{t}.{l}.writer.{h}.k"] = PartialHook(
                        f"blocks.{l}.attn.hook_k",
                        lambda x: x[:, t, h],
                        lambda x, y: replace_head_token(x, t, h, y)
                    )
                    node_layers[f"{t}.{l}.writer.{h}.k"] = l
                    # writer - v
                    nodes[f"{t}.{l}.writer.{h}.v"] = PartialHook(
                        f"blocks.{l}.attn.hook_v",
                        lambda x: x[:, t, h],
                        lambda x, y: replace_head_token(x, t, h, y)
                    )
                    node_layers[f"{t}.{l}.writer.{h}.v"] = l
                    """

                def prev_layer_resid(t, l):
                    if l == 0:
                        return f"{t}.0.resid_pre"
                    else:
                        return f"{t}.{l-1}.resid_post"                

                # edges
                resid_in = prev_layer_resid(t, l)
                if l == 0:
                    resid_in_hook = PartialHook(
                        "blocks.0.hook_resid_pre",
                        lambda x: idx_token(x, t),
                        lambda x, y: replace_token(x, t, y)
                    )
                    nodes[resid_in] = resid_in_hook
                    node_layers[resid_in] = 0
                
                new_edges = [
                    (f"{t}.{l}.resid_mid", resid_in),
                    *[(f"{t}.{l}.writer.{h}.q", resid_in) for h in range(cfg.n_heads)],
                    *[(f"{t}.{l}.writer.{h}.k", resid_in) for h in range(cfg.n_heads)],
                    *[(f"{t}.{l}.writer.{h}.v", resid_in) for h in range(cfg.n_heads)],
                    # reader depends on same writer query head, and all writer key and value heads from previous tokens
                    #*[(f"{t}.{l}.reader.{h}", f"{t}.{l}.writer.{h}.q") for h in range(cfg.n_heads)],
                    #*[(f"{t}.{l}.reader.{h}", f"{new_t}.{l}.writer.{h}.k") for h in range(cfg.n_heads) for new_t in range(t+1)],
                    #*[(f"{t}.{l}.reader.{h}", f"{new_t}.{l}.writer.{h}.v") for h in range(cfg.n_heads) for new_t in range(t+1)],
                    *[(f"{t}.{l}.reader.{h}", prev_layer_resid(new_t, l)) for h in range(cfg.n_heads) for new_t in range(t)],
                    *[(f"{t}.{l}.resid_mid", f"{t}.{l}.reader.{h}") for h in range(cfg.n_heads)],
                    (f"{t}.{l}.mlp_out", f"{t}.{l}.resid_mid"),
                    (f"{t}.{l}.resid_post", f"{t}.{l}.mlp_out"),
                    (f"{t}.{l}.resid_post", f"{t}.{l}.resid_mid"),
                ]

                edges.extend(new_edges)
        output = PartialHook(
            f"blocks.{cfg.n_layers - 1}.hook_resid_post",
            lambda x: x,
            lambda x, y: y
        )
        nodes["output"] = output
        node_layers["output"] = cfg.n_layers - 1
        for t in range(n_ctx):
            edges.append(("output", f"{t}.{cfg.n_layers - 1}.resid_post"))
        return ComputeGraph(nodes, edges, "output"), node_layers
    
class ProbedTransformer(Transformer):
    def __init__(self, transformer, cfg, n_ctx, probe_hook: PartialHook, probe_reads_from: str = "output"):
        super().__init__(transformer, cfg, n_ctx)
        
        self.graph.nodes["probe"] = probe_hook
        self.graph.edges.append(("probe", probe_reads_from))
        self.node_layers["probe"] = self.node_layers[probe_reads_from]