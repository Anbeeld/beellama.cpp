import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf
from conversion.qwen import Qwen3NextModel
from conversion.qwen4exp import Qwen4ExpTextModel


def make_model(*, mtp_only: bool, ple_layers: list[int]) -> Qwen4ExpTextModel:
    model = Qwen4ExpTextModel.__new__(Qwen4ExpTextModel)
    model.mtp_only = mtp_only
    model.no_mtp = False
    model.rope_parameters = {}
    model.gguf_writer = gguf.GGUFWriter(
        path=None, arch=gguf.MODEL_ARCH_NAMES[model.model_arch]
    )
    model.hparams = {
        "hc_count": 2,
        "hc_lowrank": 1,
        "num_hidden_layers": 2,
        "indexer_n_heads": 2,
        "indexer_head_dim": 4,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "layer_types": ["full_attention", "linear_attention"],
        "ple_layer_ids": ple_layers,
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "ple_conv_kernel_size": 4,
        "eos_token_id": 99,
    }
    model._ple_row_dim = 4
    model.model_tensors = {
        "model.ple_embedding.layer_multipliers": lambda: torch.tensor([1, 2, 3], dtype=torch.int64),
        "model.ple_embedding.ngram_heads_offsets": lambda: torch.tensor([0, 1], dtype=torch.int64),
        "model.ple_embedding.ngram_heads_vocab_sizes": lambda: torch.tensor([8, 8], dtype=torch.int64),
    }
    return model


def metadata_keys(model: Qwen4ExpTextModel) -> set[str]:
    return set(model.gguf_writer.kv_data[0])


def test_mtp_only_filters_ple_but_preserves_hyper_connection_mixer() -> None:
    with patch.object(Qwen4ExpTextModel, "_original_block_count", 2), \
            patch.object(Qwen4ExpTextModel, "mtp_only", True):
        assert Qwen4ExpTextModel.filter_tensors(
            ("model.ple_embedding.shard_0.weight", lambda: torch.zeros(1))
        ) is None
        assert Qwen4ExpTextModel.filter_tensors(
            ("model.ple_embedding.layer_multipliers", lambda: torch.zeros(1))
        ) is None
        for name in Qwen4ExpTextModel.mtp_only_extra_tensor_prefixes:
            assert Qwen4ExpTextModel.filter_tensors((name + ".weight", lambda: torch.zeros(1))) is not None


def test_mtp_only_omits_ple_metadata_but_keeps_non_ple_parameters() -> None:
    model = make_model(mtp_only=True, ple_layers=[1])
    model.model_tensors = {}  # PLE constants were removed by MTP-only filtering.
    with patch.object(Qwen3NextModel, "set_gguf_parameters"):
        model.set_gguf_parameters()

    keys = metadata_keys(model)
    assert not any("ple." in key for key in keys)
    assert any("hyper_connection.count" in key for key in keys)
    assert any("attention.compress_ratios" in key for key in keys)


def test_regular_model_retains_ple_metadata() -> None:
    model = make_model(mtp_only=False, ple_layers=[1])
    with patch.object(Qwen3NextModel, "set_gguf_parameters"):
        model.set_gguf_parameters()

    keys = metadata_keys(model)
    assert any("ple.layers" in key for key in keys)
    assert any("ple.ngram_size" in key for key in keys)
    assert any("ple.layer_multipliers" in key for key in keys)


if __name__ == "__main__":
    test_mtp_only_filters_ple_but_preserves_hyper_connection_mixer()
    test_mtp_only_omits_ple_metadata_but_keeps_non_ple_parameters()
    test_regular_model_retains_ple_metadata()
    print("Qwen4Exp MTP conversion: 3 tests passed")
