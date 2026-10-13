from types import SimpleNamespace

from slime.utils.megatron_bridge_utils import patch_hf_config_for_megatron_bridge


def test_qwen35_bridge_patch_disables_mtp_by_default(monkeypatch):
    monkeypatch.delenv("SLIME_ENABLE_QWEN35_MTP", raising=False)
    text_config = SimpleNamespace(model_type="qwen3_5_moe_text", mtp_num_hidden_layers=1)
    hf_config = SimpleNamespace(model_type="qwen3_5_moe", text_config=text_config)

    patch_hf_config_for_megatron_bridge(hf_config)

    assert text_config.mtp_num_hidden_layers is None


def test_qwen35_bridge_patch_can_keep_mtp_when_enabled(monkeypatch):
    monkeypatch.setenv("SLIME_ENABLE_QWEN35_MTP", "1")
    text_config = SimpleNamespace(model_type="qwen3_5_moe_text", mtp_num_hidden_layers=1)
    hf_config = SimpleNamespace(model_type="qwen3_5_moe", text_config=text_config)

    patch_hf_config_for_megatron_bridge(hf_config)

    assert text_config.mtp_num_hidden_layers == 1


def test_bridge_patch_does_not_disable_non_qwen35_mtp(monkeypatch):
    monkeypatch.delenv("SLIME_ENABLE_QWEN35_MTP", raising=False)
    hf_config = SimpleNamespace(model_type="mimo", mtp_num_hidden_layers=1)

    patch_hf_config_for_megatron_bridge(hf_config)

    assert hf_config.mtp_num_hidden_layers == 1
