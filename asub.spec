# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building a single-file asub.exe."""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None


def optional_metadata(package):
    try:
        return copy_metadata(package)
    except Exception:
        return []


datas = (
    collect_data_files("faster_whisper")
    + collect_data_files("whisperx")
    + collect_data_files("pyannote.audio")
    + collect_data_files("torchcodec")
    + optional_metadata("whisperx")
    + optional_metadata("pyannote.audio")
    + optional_metadata("pyannote.core")
    + optional_metadata("pyannote.database")
    + optional_metadata("pyannote.metrics")
    + optional_metadata("pyannote.pipeline")
    + optional_metadata("torch")
    + optional_metadata("torchaudio")
    + optional_metadata("torchvision")
    + optional_metadata("torchcodec")
    + optional_metadata("transformers")
    + optional_metadata("huggingface-hub")
    + optional_metadata("tokenizers")
    + optional_metadata("safetensors")
    + optional_metadata("lightning")
    + optional_metadata("pytorch-lightning")
    + optional_metadata("torchmetrics")
    + optional_metadata("omegaconf")
    + optional_metadata("pandas")
    + optional_metadata("scikit-learn")
)

hiddenimports = [
    "faster_whisper",
    "ctranslate2",
    "huggingface_hub",
    "deep_translator",
    "requests",
    "whisperx",
    "whisperx.diarize",
    "pyannote.audio",
    "torch",
    "torchaudio",
    "torchvision",
    "torchcodec",
    "transformers",
    "pandas",
    "sklearn",
    *collect_submodules("whisperx"),
    *collect_submodules("pyannote"),
]

a = Analysis(
    ["src/asub/__main__.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="asub",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
