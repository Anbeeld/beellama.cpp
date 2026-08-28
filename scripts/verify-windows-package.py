#!/usr/bin/env python3
"""Verify local PE import/export consistency in Windows release packages."""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path


class PEFormatError(Exception):
    pass


class PEFile:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        self.sections: list[tuple[int, int, int]] = []

        if len(self.data) < 0x40 or self.data[:2] != b"MZ":
            raise PEFormatError("missing MZ header")

        pe_offset = self.u32(0x3C)
        if self.data[pe_offset : pe_offset + 4] != b"PE\0\0":
            raise PEFormatError("missing PE header")

        coff_offset = pe_offset + 4
        section_count = self.u16(coff_offset + 2)
        optional_offset = coff_offset + 20
        optional_size = self.u16(coff_offset + 16)
        magic = self.u16(optional_offset)
        if magic == 0x20B:
            self.is_64_bit = True
            self.data_directories_offset = optional_offset + 112
        elif magic == 0x10B:
            self.is_64_bit = False
            self.data_directories_offset = optional_offset + 96
        else:
            raise PEFormatError(f"unsupported optional header magic 0x{magic:x}")

        section_offset = optional_offset + optional_size
        for index in range(section_count):
            offset = section_offset + index * 40
            virtual_size = self.u32(offset + 8)
            virtual_address = self.u32(offset + 12)
            raw_size = self.u32(offset + 16)
            raw_offset = self.u32(offset + 20)
            self.sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def u16(self, offset: int) -> int:
        return struct.unpack_from("<H", self.data, offset)[0]

    def u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self.data, offset)[0]

    def u64(self, offset: int) -> int:
        return struct.unpack_from("<Q", self.data, offset)[0]

    def cstring(self, offset: int) -> str:
        end = offset
        while end < len(self.data) and self.data[end] != 0:
            end += 1
        return self.data[offset:end].decode("ascii", errors="replace")

    def data_directory(self, index: int) -> tuple[int, int]:
        offset = self.data_directories_offset + index * 8
        return self.u32(offset), self.u32(offset + 4)

    def rva_to_offset(self, rva: int) -> int:
        for virtual_address, size, raw_offset in self.sections:
            if virtual_address <= rva < virtual_address + size:
                return raw_offset + (rva - virtual_address)
        raise PEFormatError(f"RVA 0x{rva:x} is outside mapped sections")

    def imports(self) -> list[tuple[str, str]]:
        import_rva, _ = self.data_directory(1)
        if import_rva == 0:
            return []

        imports: list[tuple[str, str]] = []
        descriptor_offset = self.rva_to_offset(import_rva)
        thunk_size = 8 if self.is_64_bit else 4
        ordinal_flag = 0x8000000000000000 if self.is_64_bit else 0x80000000

        while True:
            original_first_thunk = self.u32(descriptor_offset)
            name_rva = self.u32(descriptor_offset + 12)
            first_thunk = self.u32(descriptor_offset + 16)
            if original_first_thunk == 0 and name_rva == 0 and first_thunk == 0:
                break

            dll_name = self.cstring(self.rva_to_offset(name_rva)).lower()
            thunk_rva = original_first_thunk or first_thunk
            thunk_offset = self.rva_to_offset(thunk_rva)

            while True:
                thunk = self.u64(thunk_offset) if self.is_64_bit else self.u32(thunk_offset)
                if thunk == 0:
                    break

                if thunk & ordinal_flag:
                    symbol = f"#{thunk & 0xFFFF}"
                else:
                    symbol = self.cstring(self.rva_to_offset(thunk) + 2)

                imports.append((dll_name, symbol))
                thunk_offset += thunk_size

            descriptor_offset += 20

        return imports

    def exports(self) -> set[str]:
        export_rva, _ = self.data_directory(0)
        if export_rva == 0:
            return set()

        export_offset = self.rva_to_offset(export_rva)
        name_count = self.u32(export_offset + 24)
        names_rva = self.u32(export_offset + 32)
        if name_count == 0 or names_rva == 0:
            return set()

        names_offset = self.rva_to_offset(names_rva)
        exports: set[str] = set()
        for index in range(name_count):
            name_rva = self.u32(names_offset + index * 4)
            exports.add(self.cstring(self.rva_to_offset(name_rva)))
        return exports


def iter_package_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dll", ".exe"}
    ]


def expected_file_failures(
    root: Path,
    required_names: list[str],
    required_globs: list[str],
    forbidden_names: list[str],
) -> list[str]:
    names = [path.name.casefold() for path in root.rglob("*") if path.is_file()]
    failures: list[str] = []

    for required in required_names:
        if required.casefold() not in names:
            failures.append(f"missing required file {required}")
    for pattern in required_globs:
        if not any(fnmatch.fnmatchcase(name, pattern.casefold()) for name in names):
            failures.append(f"missing required file matching {pattern}")
    for forbidden in forbidden_names:
        if forbidden.casefold() in names:
            failures.append(f"forbidden file present: {forbidden}")

    return failures


def required_local_import_failures(
    external_imports: set[str],
    required_local_imports: list[str],
) -> list[str]:
    external_names = {name.casefold() for name in external_imports}
    return [
        f"required local import is missing: {required}"
        for required in required_local_imports
        if required.casefold() in external_names
    ]


def duplicate_archive_entries(path: Path) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/").lstrip("/").casefold()
            if normalized in seen:
                duplicates.append(info.filename)
            else:
                seen.add(normalized)
    return duplicates


def verify_directory(
    root: Path,
    verbose: bool,
    required_names: list[str],
    required_globs: list[str],
    forbidden_names: list[str],
    required_local_imports: list[str],
) -> int:
    binaries = iter_package_files(root)
    by_name: dict[str, list[Path]] = {}
    for binary in binaries:
        by_name.setdefault(binary.name.lower(), []).append(binary)

    export_cache: dict[Path, set[str]] = {}
    failures = expected_file_failures(
        root,
        required_names=required_names,
        required_globs=required_globs,
        forbidden_names=forbidden_names,
    )
    local_edges = 0
    external_imports: set[str] = set()

    for binary in binaries:
        try:
            pe = PEFile(binary)
            imports = pe.imports()
        except (OSError, PEFormatError, struct.error) as exc:
            failures.append(f"{binary.relative_to(root)}: cannot parse PE file: {exc}")
            continue

        for dll_name, symbol in imports:
            local_matches = by_name.get(dll_name)
            if not local_matches:
                external_imports.add(dll_name)
                continue

            dependency = sorted(
                local_matches,
                key=lambda path: (len(path.relative_to(root).parts), str(path).lower()),
            )[0]
            local_edges += 1

            if symbol.startswith("#"):
                continue

            if dependency not in export_cache:
                try:
                    export_cache[dependency] = PEFile(dependency).exports()
                except (OSError, PEFormatError, struct.error) as exc:
                    failures.append(
                        f"{binary.relative_to(root)} -> {dependency.relative_to(root)}: "
                        f"cannot parse dependency exports: {exc}"
                    )
                    export_cache[dependency] = set()

            if symbol not in export_cache[dependency]:
                failures.append(
                    f"{binary.relative_to(root)} -> {dependency.relative_to(root)}: "
                    f"missing import {symbol}"
                )

    failures.extend(
        required_local_import_failures(
            external_imports=external_imports,
            required_local_imports=required_local_imports,
        )
    )

    if verbose and external_imports:
        print("External imports not checked:")
        for dll_name in sorted(external_imports):
            print(f"  {dll_name}")

    if failures:
        print(f"Windows package verification failed for {root}:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"Windows package verification passed for {root}: "
        f"{len(binaries)} PE files, {local_edges} local imports checked"
    )
    return 0


def verify_path(
    path: Path,
    verbose: bool,
    required_names: list[str],
    required_globs: list[str],
    forbidden_names: list[str],
    required_local_imports: list[str],
) -> int:
    if path.is_dir():
        return verify_directory(
            path,
            verbose,
            required_names=required_names,
            required_globs=required_globs,
            forbidden_names=forbidden_names,
            required_local_imports=required_local_imports,
        )

    if path.suffix.lower() != ".zip":
        print(f"{path}: expected a directory or .zip file", file=sys.stderr)
        return 1

    duplicates = duplicate_archive_entries(path)
    if duplicates:
        print(f"Windows package verification failed for {path}:", file=sys.stderr)
        for duplicate in duplicates:
            print(f"  duplicate ZIP entry: {duplicate}", file=sys.stderr)
        return 1

    temp_dir = Path(tempfile.mkdtemp(prefix="verify-windows-package-"))
    try:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temp_dir)
        return verify_directory(
            temp_dir,
            verbose,
            required_names=required_names,
            required_globs=required_globs,
            forbidden_names=forbidden_names,
            required_local_imports=required_local_imports,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--require-name", action="append", default=[])
    parser.add_argument("--require-glob", action="append", default=[])
    parser.add_argument("--forbid-name", action="append", default=[])
    parser.add_argument("--require-local-import", action="append", default=[])
    args = parser.parse_args()

    status = 0
    for path in args.paths:
        if not path.exists():
            print(f"{path}: path does not exist", file=sys.stderr)
            status = 1
            continue
        status = verify_path(
            path,
            args.verbose,
            required_names=args.require_name,
            required_globs=args.require_glob,
            forbidden_names=args.forbid_name,
            required_local_imports=args.require_local_import,
        ) or status
    return status


if __name__ == "__main__":
    raise SystemExit(main())
