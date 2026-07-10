from __future__ import annotations

import subprocess
import sys
import tempfile
import gc
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402


def make_legacy_draft(path: Path, canonical_mask: int | None = None) -> None:
    writer = gguf.GGUFWriter(path, "dflash-draft")
    writer.add_uint32("dflash-draft.block_count", 1)
    writer.add_uint32("dflash-draft.embedding_length", 4)
    writer.add_array("dflash-draft.dflash.target_layer_ids", [1, 3])
    writer.add_uint32("dflash-draft.dflash.block_size", 16)
    writer.add_uint32("dflash-draft.dflash.mask_token_id", 42)
    if canonical_mask is not None:
        writer.add_uint32(gguf.Keys.Tokenizer.MASK_ID, canonical_mask)
    writer.add_tensor("dflash_fc.weight", np.arange(32, dtype=np.float32).reshape(4, 8))
    writer.add_tensor("dflash_hidden_norm.weight", np.ones((4,), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.full((4,), 2.0, dtype=np.float32))
    writer.add_tensor("blk.0.post_attention_norm.weight", np.full((4,), 3.0, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def field(reader: gguf.GGUFReader, key: str):
    return reader.fields[key].contents()


def main() -> None:
    converter = ROOT / "scripts" / "convert-dflash-draft-to-upstream.py"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        source = temp / "legacy.gguf"
        output = temp / "upstream.gguf"
        # Some legacy drafts (including Gemma 4) already contain the canonical
        # tokenizer mask key in addition to the architecture-local copy.
        make_legacy_draft(source, canonical_mask=42)

        subprocess.run([sys.executable, str(converter), str(source), str(output), "--verify"], check=True)
        reader = gguf.GGUFReader(output, "r")

        assert field(reader, "general.architecture") == "dflash"
        assert field(reader, "dflash.target_layers") == [1, 3]
        assert field(reader, "dflash.block_size") == 16
        assert field(reader, gguf.Keys.Tokenizer.MASK_ID) == 42
        assert "dflash.mask_token_id" not in reader.fields
        assert "dflash-draft.block_count" not in reader.fields
        assert field(reader, "dflash.block_count") == 1
        assert {tensor.name for tensor in reader.tensors} == {
            "fc.weight",
            "enc.output_norm.weight",
            "output_norm.weight",
            "blk.0.ffn_norm.weight",
        }

        output_tensors = {tensor.name: tensor for tensor in reader.tensors}
        expected = {
            "fc.weight": np.arange(32, dtype=np.float32).reshape(4, 8),
            "enc.output_norm.weight": np.ones((4,), dtype=np.float32),
            "output_norm.weight": np.full((4,), 2.0, dtype=np.float32),
            "blk.0.ffn_norm.weight": np.full((4,), 3.0, dtype=np.float32),
        }
        for name, values in expected.items():
            assert output_tensors[name].tensor_type == gguf.GGMLQuantizationType.F32
            assert np.array_equal(output_tensors[name].data, values)
        del output_tensors
        del reader
        gc.collect()

        conflicting_source = temp / "legacy-conflicting-mask.gguf"
        conflicting_output = temp / "upstream-conflicting-mask.gguf"
        make_legacy_draft(conflicting_source, canonical_mask=41)
        conflict = subprocess.run(
            [sys.executable, str(converter), str(conflicting_source), str(conflicting_output), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert conflict.returncode == 2
        assert "conflicting metadata values" in conflict.stderr


if __name__ == "__main__":
    main()
