from contextlib import contextmanager
import os

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


_TRUE_VALUES = {"1", "true", "yes", "on"}

def _is_true_env(name):
    return os.environ.get(name, "").lower() in _TRUE_VALUES


def _is_qwen35_config(config):
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str) and model_type.startswith("qwen3_5"):
        return True

    names = [type(config).__name__]
    architectures = getattr(config, "architectures", None) or []
    names.extend(architectures)
    return any("qwen3_5" in str(name).lower() or "qwen35" in str(name).lower() for name in names)


def _patch_qwen35_mtp_for_bridge(configs):
    if _is_true_env("SLIME_ENABLE_QWEN35_MTP"):
        return

    if not any(_is_qwen35_config(config) for config in configs):
        return

    for config in configs:
        if hasattr(config, "mtp_num_hidden_layers"):
            config.mtp_num_hidden_layers = None


def patch_hf_config_for_megatron_bridge(hf_config):
    configs = []
    seen_config_ids = set()

    def add_config(config):
        if config is None or id(config) in seen_config_ids:
            return
        seen_config_ids.add(id(config))
        configs.append(config)

    add_config(hf_config)
    add_config(getattr(hf_config, "config", None))

    for config in list(configs):
        add_config(getattr(config, "text_config", None))

    for config in configs:
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params and not hasattr(config, "rope_theta"):
            config.rope_theta = rope_params["rope_theta"]

    _patch_qwen35_mtp_for_bridge(configs)

    return hf_config


def patch_auto_bridge_hf_config(bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        patch_hf_config_for_megatron_bridge(hf_pretrained)

    return bridge


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
