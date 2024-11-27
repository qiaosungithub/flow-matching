import jax.numpy as jnp
from collections import defaultdict

# class LastItem:
#     def __init__(self):
#         self.val = 114514.1919810
#     def append(self, v):
#         self.val = v
#     def get(self):
#         return self.val

# class Avger(list):
#     def get(self):
#         return sum(self) / len(self) if len(self) > 0 else 114514.1919810

# class MyMetrics:
#     def __init__(self, reduction=Avger):
#         self.reduction = reduction
#         self.metrics = defaultdict(reduction)

#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             self.metrics[k].append(v)

#     def compute(self):
#         return {k: v.get() for k, v in self.metrics.items()}

#     def reset(self):
#         self.metrics = defaultdict(self.reduction)

def tang_reduce(d:dict):
    for k, v in d.items():
        if k in ("label", "labels"): continue
        d[k] = jnp.mean(v)