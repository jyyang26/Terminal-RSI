"""Shared pytest setup for the CPU test suite.

Import the real ``torch`` before any test module is collected.

Several Harbor tests install a stub ``torch`` module so they can run without the
real package::

    # tests/test_harbor_atif_adapter.py
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        ...
        sys.modules["torch"] = torch_stub

The guard is order-dependent: whichever module is imported first wins. When a
stubbing module is collected before a module that needs the real package, the
later import fails with ``No module named 'torch.nn'; 'torch' is not a
package``. That is how ``tests/test_qwen3_linear_attention_cu_seqlens.py`` ends
up failing at collection in a full-suite run while passing on its own, and the
same stub also breaks ``import sglang_router`` inside
``slime.utils.arguments``.

Importing torch here makes the guards no-ops, so every module sees the real
package and collection order stops mattering. The stubs remain useful in
environments where torch genuinely is absent -- there the import below fails
softly and the guards fire exactly as before.
"""

try:  # pragma: no cover - environment dependent
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - torch-less environments keep the stubs
    pass
