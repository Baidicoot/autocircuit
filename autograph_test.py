# %%
import autocircuit as ac
import autocircuit.visualise as vis
import autocircuit.transformer_lens as ac_lens

import transformer_lens as lens
import torch

# %%
cfg = lens.HookedTransformerConfig(8, 512, 59, 64, d_vocab=61, act_fn="relu")
model = lens.HookedTransformer(cfg)

# %%
graph = ac_lens.GraphedHookedTransformer.build_graph(model, cfg, 4)

print([edge for edge in graph.edges if edge[0] == "2.0.reader.0"])

# %%
vis.draw_graph(graph, cfg)
