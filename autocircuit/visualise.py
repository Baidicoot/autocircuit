import networkx
import matplotlib.pyplot as plt
from .graphed_model import ComputeGraph, PartialHook

def calc_positions(cfg, nodes=None):
    pos = {"output": (0, cfg.n_layers)}
    for t in range(cfg.n_ctx):
        t_col_size = 2 / cfg.n_ctx
        t_col_offset = t_col_size * t
        for l in range(cfg.n_layers):
            pos[f"{t}.{l}.resid_post"] = (t_col_offset + t_col_size / 2, l)
            pos[f"{t}.{l}.resid_mid"] = (t_col_offset + t_col_size / 2, l - 0.25)
            pos[f"{t}.{l}.mlp_out"] = (t_col_offset, l - 0.125)
            for h in range(cfg.n_heads):
                rel_h = h + 0.5
                pos[f"{t}.{l}.reader.{h}"] = (t_col_offset + t_col_size * rel_h / cfg.n_heads, l - 0.4)
                #pos[f"{t}.{l}.writer.{h}.q"] = (t_col_offset + t_col_size * (rel_h-0.25) / cfg.n_heads, l - 0.8)
                #pos[f"{t}.{l}.writer.{h}.k"] = (t_col_offset + t_col_size * rel_h / cfg.n_heads, l - 0.8)
                #pos[f"{t}.{l}.writer.{h}.v"] = (t_col_offset + t_col_size * (rel_h+0.25) / cfg.n_heads, l - 0.8)
        pos[f"{t}.0.resid_pre"] = (t_col_offset + t_col_size / 2, -1)
    if nodes is not None:
        for node in nodes:
            if node not in pos:
                pos[node] = (0, 0)
    return pos

def draw_graph(graph: ComputeGraph, cfg):
    G = networkx.DiGraph()
    for node in graph.nodes.keys():
        G.add_node(node)
    for edge in graph.edges:
        G.add_edge(edge[0], edge[1])
    pos = calc_positions(cfg, graph.nodes.keys())
    networkx.draw(G, pos=pos, with_labels=True)
    plt.show()