"""Minimal torch.nn stub for import compatibility.

Provides Module base class so that `issubclass(x, torch.nn.Module)`
checks in libraries like `datasets` (HuggingFace) don't crash.
"""


class Module:
    """Stub nn.Module — never instantiated, only used for isinstance/issubclass checks."""

    pass
