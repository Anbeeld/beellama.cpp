#!/usr/bin/env python3
"""Convert legacy BeeLlama DFlash draft GGUFs to upstream llama.cpp DFlash.

The converter rewrites only the DFlash architecture namespace, metadata names,
and tensor names. Tensor bytes and quantization types are copied verbatim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "gguf-py"))

import gguf  # noqa: E402


SUPPORTED_INPUT_ARCHITECTURES = {"dflash", "dflash-draft"}

METADATA_RENAMES = {
    "dflash.target_layer_ids": "dflash.target_layers",
    "dflash-draft.target_layer_ids": "dflash.target_layers",
    "dflash-draft.dflash.target_layer_ids": "dflash.target_layers",
    "dflash.dflash.target_layer_ids": "dflash.target_layers",
    "dflash-draft.dflash.block_size": "dflash.block_size",
    "dflash.dflash.block_size": "dflash.block_size",
    # Upstream DFlash gets the mask token from the draft vocabulary, not from
    # architecture metadata.  Promote every legacy spelling to the canonical
    # tokenizer key consumed by llama_vocab_mask().
    "dflash-draft.dflash.mask_token_id": gguf.Keys.Tokenizer.MASK_ID,
    "dflash.dflash.mask_token_id": gguf.Keys.Tokenizer.MASK_ID,
    "dflash.mask_token_id": gguf.Keys.Tokenizer.MASK_ID,
    "dflash-draft.dflash.n_target_features": "dflash.n_target_features",
    "dflash.dflash.n_target_features": "dflash.n_target_features",
}

TENSOR_RENAMES = {
    "dflash_fc.weight": "fc.weight",
    "dflash_hidden_norm.weight": "enc.output_norm.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
}


def field_value(field: gguf.ReaderField):
    return field.contents()


def metadata_name(name: str) -> str:
    if name in METADATA_RENAMES:
        return METADATA_RENAMES[name]
    if name.startswith("dflash-draft."):
        return "dflash." + name[len("dflash-draft.") :]
    return name


def tensor_name(name: str) -> str:
    if name in TENSOR_RENAMES:
        return TENSOR_RENAMES[name]
    if name.startswith("blk.") and name.endswith(".post_attention_norm.weight"):
        return name[: -len(".post_attention_norm.weight")] + ".ffn_norm.weight"
    return name


def metadata_values_equal(left, right) -> bool:
    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            metadata_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)


def mapped_metadata(reader: gguf.GGUFReader) -> dict[str, tuple[str, gguf.ReaderField]]:
    result: dict[str, tuple[str, gguf.ReaderField]] = {}
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == gguf.Keys.General.ARCHITECTURE or name.startswith("split."):
            continue

        renamed = metadata_name(name)
        previous = result.get(renamed)
        if previous is None:
            result[renamed] = (name, field)
            continue

        previous_name, previous_field = previous
        if not metadata_values_equal(field_value(previous_field), field_value(field)):
            raise ValueError(
                f"conflicting metadata values for {renamed!r} "
                f"from {previous_name!r} and {name!r}"
            )

        # Prefer a field that was canonical in the input so its declared GGUF
        # type is retained when an equal legacy alias is also present.
        if name == renamed and previous_name != renamed:
            result[renamed] = (name, field)

    return result


def architecture(reader: gguf.GGUFReader) -> str:
    field = reader.fields.get(gguf.Keys.General.ARCHITECTURE)
    if field is None:
        raise ValueError("input GGUF has no general.architecture metadata")
    value = field_value(field)
    if not isinstance(value, str):
        raise ValueError("input general.architecture metadata is not a string")
    return value


def ensure_single_file(reader: gguf.GGUFReader) -> None:
    split_count = reader.fields.get(gguf.Keys.Split.LLM_KV_SPLIT_COUNT)
    if split_count is not None and int(field_value(split_count)) != 1:
        raise ValueError("split GGUF input is not supported; merge it before conversion")


def validate_mapping(reader: gguf.GGUFReader) -> None:
    input_arch = architecture(reader)
    if input_arch not in SUPPORTED_INPUT_ARCHITECTURES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_ARCHITECTURES))
        raise ValueError(f"expected a legacy DFlash GGUF ({supported}), got {input_arch!r}")

    ensure_single_file(reader)

    output_metadata_names = set(mapped_metadata(reader))

    if "dflash.target_layers" not in output_metadata_names:
        raise ValueError("input GGUF is missing DFlash target_layer_ids metadata")
    if gguf.Keys.Tokenizer.MASK_ID not in output_metadata_names:
        raise ValueError("input GGUF is missing the DFlash mask token id")

    output_tensor_names: set[str] = set()
    for tensor in reader.tensors:
        renamed = tensor_name(tensor.name)
        if renamed in output_tensor_names:
            raise ValueError(f"tensor rename collision at {renamed!r}")
        output_tensor_names.add(renamed)

    required_tensors = {"fc.weight", "enc.output_norm.weight", "output_norm.weight"}
    missing = sorted(required_tensors - output_tensor_names)
    if missing:
        raise ValueError("input GGUF is missing required DFlash tensors after rename: " + ", ".join(missing))


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for output_name, (_source_name, field) in mapped_metadata(reader).items():
        if output_name == gguf.Keys.General.ALIGNMENT:
            continue

        value = field_value(field)
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(output_name, value, value_type, sub_type=sub_type)


def tensor_digest(tensor: gguf.ReaderTensor) -> str:
    return hashlib.sha256(tensor.data.data).hexdigest()


def verify_tensor_payloads(reader: gguf.GGUFReader, output_path: Path) -> None:
    output_reader = gguf.GGUFReader(output_path, "r")
    expected = {tensor_name(tensor.name): tensor for tensor in reader.tensors}
    actual = {tensor.name: tensor for tensor in output_reader.tensors}

    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"converted tensor set mismatch (missing={missing}, extra={extra})")

    for name, source in expected.items():
        converted = actual[name]
        if converted.tensor_type != source.tensor_type:
            raise ValueError(
                f"converted tensor type mismatch for {name}: "
                f"{source.tensor_type.name} -> {converted.tensor_type.name}"
            )
        if tensor_digest(converted) != tensor_digest(source):
            raise ValueError(f"converted tensor checksum mismatch for {name}")

    del actual
    del expected
    del output_reader
    gc.collect()


def convert(input_path: Path, output_path: Path, dry_run: bool, verify: bool) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists() and not dry_run:
        raise ValueError(f"output already exists: {output_path}")

    reader = gguf.GGUFReader(input_path, "r")
    validate_mapping(reader)

    print(f"input architecture: {architecture(reader)}")
    print(f"metadata fields: {len(reader.fields)}, tensors: {len(reader.tensors)}")
    if dry_run:
        print("validation succeeded (dry run; no output written)")
        return

    writer = gguf.GGUFWriter(output_path, "dflash", endianess=reader.endianess)
    if reader.alignment != gguf.GGUF_DEFAULT_ALIGNMENT:
        writer.add_custom_alignment(reader.alignment)

    copy_metadata(reader, writer)
    for tensor in reader.tensors:
        writer.add_tensor(
            tensor_name(tensor.name),
            tensor.data,
            raw_shape=tensor.data.shape,
            raw_dtype=tensor.tensor_type,
            tensor_endianess=reader.endianess,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote upstream DFlash GGUF: {output_path}")
    if verify:
        verify_tensor_payloads(reader, output_path)
        print("verified converted tensor types and SHA-256 payload checksums")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="legacy BeeLlama DFlash GGUF")
    parser.add_argument("output", type=Path, help="new upstream-compatible DFlash GGUF")
    parser.add_argument("--dry-run", action="store_true", help="validate conversion without writing output")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-open the output and verify every tensor type and payload checksum",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.dry_run and args.verify:
            raise ValueError("--dry-run and --verify cannot be used together")
        convert(args.input, args.output, args.dry_run, args.verify)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
