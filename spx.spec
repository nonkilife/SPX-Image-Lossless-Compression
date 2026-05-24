# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for SPX v1.0.0
#
# Bundles the entire spx package — including the Rust extension (spx_rans.pyd)
# and the rANS template data (rans_mode.npz) — into a single portable .exe.
#
# Build:
#   .venv\Scripts\pyinstaller.exe spx.spec
#
# Output: dist\spx.exe

from PyInstaller.utils.hooks import collect_all, copy_metadata

# collect_all picks up all Python modules, compiled extensions (.pyd), and
# data files declared in the package — including spx_rans.pyd and rans_mode.npz.
spx_datas, spx_binaries, spx_hiddenimports = collect_all('spx')

# copy_metadata is required so importlib.metadata.version() works inside the
# frozen exe (env.py calls it to validate numpy / zstandard / Pillow).
meta_datas = (
    copy_metadata('numpy')
    + copy_metadata('zstandard')
    + copy_metadata('Pillow')
)

# Manually include the data file and Rust extension that collect_all missed
# (it warns "not a package" when spx is only on sys.path, not site-packages).
extra_datas = [('spx/rans_mode.npz', 'spx')]
extra_binaries = [('spx/spx_rans.pyd', 'spx')]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=spx_binaries + extra_binaries,
    datas=spx_datas + meta_datas + extra_datas,
    hiddenimports=spx_hiddenimports + [
        'spx',
        'spx.codec',
        'spx.common',
        'spx.compress',
        'spx.decompress',
        'spx.env',
        'spx.predictor',
        'spx.rans',
        'spx.rans_bitplane',
        'spx.rans_selector',
        'spx.sharding',
        'spx.transform',
        'spx.test_suite',
        'spx.spx_rans',
        'zstandard',
        'PIL',
        'PIL.Image',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='spx',
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
