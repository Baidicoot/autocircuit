# autocircuit for TransformerLens

from typing import Callable, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_lens as lens
import einops

from autocircuit.graphed_model import GraphedModel, ComputeGraph, Hook, HookManager

def idx_token(tensor, t):
    return tensor[:, t]

# probably fine to do this in-place?
def replace_token(tensor, t, replacement):
    tensor[:, t] = replacement
    return tensor

def replace_head_token(tensor, t, h, replacement):
    tensor[:, t, h] = replacement
    return tensor

# thing representing some (contiguous) selection of activations in a normal transformer_lens hook
class TLHook(Hook):
    def __init__(self, hook: lens.hook_points.HookPoint, read: Callable[[torch.Tensor], torch.Tensor], write: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.hook = hook
        self.read = read
        self.write = write
    
    def add_hook(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.hook.add_hook(lambda x, hook=None: self.write(x, fn(self.read(x))))

class TLHookManager(HookManager):
    def __init__(self, hooks: Dict[str, TLHook], node_layers: Dict[str, int]):
        self._hooks = hooks
        self.node_layers = node_layers
    
    @property
    def hooks(self):
        return self._hooks

    def get_hook_layer(self, node: str) -> int:
        return self.node_layers[node]

    # could be less aggressive & redundant here, but eh
    def remove_hooks(self):
        for hook in self.hooks:
            try:
                self.hooks[hook].hook.remove_hooks()
            except:
                print(hook)

def build_graph(model: lens.HookedTransformer, n_ctx=None) -> Tuple[ComputeGraph, HookManager]:
    cfg = model.cfg
    if n_ctx is None:
        n_ctx = cfg.n_ctx
    
    nodes = []
    edges = []
    node_layers = {}
    hooks = {}
    for t in range(n_ctx):
        for l in range(cfg.n_layers):
            # hook_resid_out
            hooks[f"{t}.{l}.resid_post"] = TLHook(
                model.hook_dict[f"blocks.{l}.hook_resid_post"],
                lambda x: idx_token(x, t),
                lambda x, y: replace_token(x, t, y)
            )
            nodes.append(f"{t}.{l}.resid_post")
            node_layers[f"{t}.{l}.resid_post"] = l + 1
            # hook_resid_mid
            hooks[f"{t}.{l}.resid_mid"] = TLHook(
                model.hook_dict[f"blocks.{l}.hook_resid_mid"],
                lambda x: idx_token(x, t),
                lambda x, y: replace_token(x, t, y)
            )
            nodes.append(f"{t}.{l}.resid_mid")
            node_layers[f"{t}.{l}.resid_mid"] = l + 1
            # hook_mlp_out
            hooks[f"{t}.{l}.mlp_out"] = TLHook(
                model.hook_dict[f"blocks.{l}.hook_mlp_out"],
                lambda x: idx_token(x, t),
                lambda x, y: replace_token(x, t, y)
            )
            nodes.append(f"{t}.{l}.mlp_out")
            node_layers[f"{t}.{l}.mlp_out"] = l + 1
            # reader heads
            for h in range(cfg.n_heads):
                # reader
                hooks[f"{t}.{l}.reader.{h}"] = TLHook(
                    model.hook_dict[f"blocks.{l}.hook_attn_out"],
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
                nodes.append(f"{t}.{l}.reader.{h}")
                node_layers[f"{t}.{l}.reader.{h}"] = l + 1
                """
                # writer - q
                nodes[f"{t}.{l}.writer.{h}.q"] = TLHook(
                    f"blocks.{l}.attn.hook_q",
                    lambda x: x[:, t, h],
                    lambda x, y: replace_head_token(x, t, h, y)
                )
                node_layers[f"{t}.{l}.writer.{h}.q"] = l
                # writer - k
                nodes[f"{t}.{l}.writer.{h}.k"] = TLHook(
                    f"blocks.{l}.attn.hook_k",
                    lambda x: x[:, t, h],
                    lambda x, y: replace_head_token(x, t, h, y)
                )
                node_layers[f"{t}.{l}.writer.{h}.k"] = l
                # writer - v
                nodes[f"{t}.{l}.writer.{h}.v"] = TLHook(
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
                resid_in_hook = TLHook(
                    model.hook_dict["blocks.0.hook_resid_pre"],
                    lambda x: idx_token(x, t),
                    lambda x, y: replace_token(x, t, y)
                )
                nodes.append(resid_in)
                hooks[resid_in] = resid_in_hook
                node_layers[resid_in] = 1
            
            new_edges = [
                (f"{t}.{l}.resid_mid", resid_in),
                #*[(f"{t}.{l}.writer.{h}.q", resid_in) for h in range(cfg.n_heads)],
                #*[(f"{t}.{l}.writer.{h}.k", resid_in) for h in range(cfg.n_heads)],
                #*[(f"{t}.{l}.writer.{h}.v", resid_in) for h in range(cfg.n_heads)],
                # reader depends on same writer query head, and all writer key and value heads from previous tokens
                #*[(f"{t}.{l}.reader.{h}", f"{t}.{l}.writer.{h}.q") for h in range(cfg.n_heads)],
                #*[(f"{t}.{l}.reader.{h}", f"{new_t}.{l}.writer.{h}.k") for h in range(cfg.n_heads) for new_t in range(t+1)],
                #*[(f"{t}.{l}.reader.{h}", f"{new_t}.{l}.writer.{h}.v") for h in range(cfg.n_heads) for new_t in range(t+1)],
                *[(f"{t}.{l}.reader.{h}", prev_layer_resid(new_t, l)) for h in range(cfg.n_heads) for new_t in range(t+1)],
                *[(f"{t}.{l}.resid_mid", f"{t}.{l}.reader.{h}") for h in range(cfg.n_heads)],
                (f"{t}.{l}.mlp_out", f"{t}.{l}.resid_mid"),
                (f"{t}.{l}.resid_post", f"{t}.{l}.mlp_out"),
                (f"{t}.{l}.resid_post", f"{t}.{l}.resid_mid"),
            ]

            edges.extend(new_edges)
    output = TLHook(
        model.hook_dict[f"blocks.{cfg.n_layers - 1}.hook_resid_post"],
        lambda x: x,
        lambda x, y: y
    )
    nodes.append("output")
    hooks["output"] = output
    node_layers["output"] = cfg.n_layers
    for t in range(n_ctx):
        edges.append(("output", f"{t}.{cfg.n_layers - 1}.resid_post"))
    return ComputeGraph(nodes, edges, "output"), TLHookManager(hooks, node_layers)