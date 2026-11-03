"""Content identity for the complete local SINQ Python implementation closure."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys


_REQUIRED_DEPENDENCIES = (
    'numpy', 'torch', 'transformers', 'safetensors', 'gguf', 'huggingface-hub',
)
_TREES = ('tools/sinq/research', 'gguf-py/gguf', 'conversion')
_FILES = ('tools/sinq/export_activations.py', 'convert_hf_to_gguf.py')


def _checked(root, path):
    # Reject links even when they currently point inside the repository: their
    # traversal and target can otherwise change independently of the manifest.
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('implementation path must not be a link: ' + relative.as_posix())
    if not path.resolve().is_relative_to(root):
        raise ValueError('implementation path escapes repository: ' + relative.as_posix())


def _sources(root, directory, *, research=False):
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == '__pycache__' or path.suffix in ('.pyc', '.pyo'):
            continue
        _checked(root, path)
        if path.is_dir():
            yield from _sources(root, path, research=research)
        elif path.is_file():
            # Include all Python code and typing metadata, not checkpoints or
            # arbitrary data that happen to live alongside implementation code.
            if path.suffix == '.py' or (not research and (
                    path.suffix == '.pyi' or path.name == 'py.typed')):
                yield path
        else:
            raise ValueError('implementation path is not a regular file: ' + path.relative_to(root).as_posix())


def _versions():
    # All installed distribution identities are a conservative strict bound on
    # direct, transitive and optional/lazy dependencies. Read metadata only: no
    # package imports, dependency resolution, installation, or network access.
    dependencies = dict.fromkeys(_REQUIRED_DEPENDENCIES)
    for distribution in metadata.distributions():
        name = distribution.metadata.get('Name')
        version = distribution.version
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
            raise ValueError('invalid installed distribution name')
        if not isinstance(version, str) or not version or any(c in version for c in ('/', '\\', '\n', '\r')):
            raise ValueError('invalid installed distribution version for ' + name)
        name = re.sub(r'[-_.]+', '-', name).lower()
        existing = dependencies.get(name)
        if existing is not None and existing != version:
            raise ValueError('conflicting installed distribution versions for ' + name)
        dependencies[name] = version
    return {
        'python_implementation': platform.python_implementation(),
        'python_version': platform.python_version(),
        'python_version_info': list(sys.version_info),
        'platform': platform.system(),
        'machine': platform.machine(),
        'dependencies': dict(sorted(dependencies.items())),
    }


def native_runtime_fingerprint(executables):
    """Conservatively bind local shared libraries alongside configured binaries.

    Include bin and sibling lib/lib64 trees; only relative names and content
    hashes are persisted. This is not a system-loader attestation.
    """
    result={}
    for role,value in sorted(executables.items()):
        executable=Path(value).resolve()
        if not executable.is_file():raise ValueError('native executable unavailable')
        def file_hash(path):
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
            return h.hexdigest()
        files={'executable':file_hash(executable)}
        for label,directory in [('bin',executable.parent),('lib',executable.parent.parent/'lib'),('lib64',executable.parent.parent/'lib64')]:
            if not directory.is_dir():continue
            for path in sorted(directory.iterdir()):
                if not path.is_file() or not (path.suffix.lower() in ('.dll','.so','.dylib') or '.so.' in path.name):continue
                if label!='bin' and not path.name.startswith(('ggml','llama','libggml','libllama')):continue
                if not path.resolve().is_relative_to(executable.parent.parent):raise ValueError('native dependency escapes configured prefix')
                files[label+'/'+path.name]=file_hash(path)
        result[role]=files
    return result


def implementation_fingerprint(root=None):
    """Return a versioned manifest and SHA-256 of its canonical JSON identity.

    ``root`` is an optional repository-shaped fixture directory. Every call
    reads source bytes and installed metadata afresh, including when mtimes
    are unchanged. Required source trees and entrypoints fail closed.
    """
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]
    paths = []
    for name in _TREES:
        directory = root / name
        _checked(root, directory)
        if not directory.is_dir():
            raise ValueError('missing implementation tree: ' + name)
        sources = list(_sources(root, directory, research=name == 'tools/sinq/research'))
        if not any(path.suffix == '.py' for path in sources):
            raise ValueError('empty implementation tree: ' + name)
        paths.extend(sources)
    for name in _FILES:
        path = root / name
        _checked(root, path)
        if not path.is_file():
            raise ValueError('missing implementation file: ' + name)
        paths.append(path)
    files = {}
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        files[path.relative_to(root).as_posix()] = digest.hexdigest()
    result = {'schema_version': 1, 'files': files, 'versions': _versions()}
    canonical = json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    result['checksum'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return result
