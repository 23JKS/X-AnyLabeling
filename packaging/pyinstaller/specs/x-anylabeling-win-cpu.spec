# -*- mode: python -*-
# vim: ft=python

import importlib.util
import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

sys.setrecursionlimit(5000)  # required on Windows

def _resolve_root_dir():
    env_root = os.environ.get('X_ANYLABELING_ROOT')
    if env_root:
        return os.path.abspath(env_root)

    spec_path = globals().get('SPEC')
    if isinstance(spec_path, str) and spec_path:
        return os.path.abspath(os.path.join(os.path.dirname(spec_path), '..', '..', '..'))

    return os.path.abspath(os.getcwd())

ROOT_DIR = _resolve_root_dir()

def _p(*parts):
    return os.path.join(ROOT_DIR, *parts)

def _load_version():
    app_info_path = _p('anylabeling', 'app_info.py')
    with open(app_info_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Failed to read __version__ from: {app_info_path}")
    return match.group(1)

__version__ = _load_version()

MSVC_RUNTIME_DLLS = {
    'msvcp140.dll',
    'msvcp140_1.dll',
    'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll',
    'concrt140.dll',
    'vcomp140.dll',
}

def _entry_dll_names(entry):
    names = []
    if isinstance(entry, (tuple, list)):
        for value in entry[:2]:
            if isinstance(value, str):
                names.append(os.path.basename(value).lower())
    return names

def _collect_onnxruntime_dlls():
    ort_spec = importlib.util.find_spec('onnxruntime')
    if ort_spec is None or ort_spec.origin is None:
        return []

    ort_capi = os.path.join(os.path.dirname(ort_spec.origin), 'capi')
    if not os.path.isdir(ort_capi):
        return []

    ort_dlls = [
        os.path.join(ort_capi, filename)
        for filename in os.listdir(ort_capi)
        if filename.lower().endswith('.dll')
    ]
    root_copy_names = (
        'onnxruntime_providers_shared.dll',
        'onnxruntime.dll',
    )
    root_copy_dlls = [
        os.path.join(ort_capi, name)
        for name in root_copy_names
        if os.path.isfile(os.path.join(ort_capi, name))
    ]
    return (
        [(dll, 'onnxruntime/capi') for dll in ort_dlls]
        + [(dll, '.') for dll in root_copy_dlls]
    )

def _collect_msvc_runtime_dlls():
    system_root = os.environ.get('SystemRoot') or os.environ.get('WINDIR')
    candidate_dirs = []
    if system_root:
        candidate_dirs.append(os.path.join(system_root, 'System32'))
    candidate_dirs.append(os.path.dirname(sys.executable))
    candidate_dirs.append(sys.base_prefix)
    candidate_dirs.extend(os.environ.get('PATH', '').split(os.pathsep))

    search_dirs = []
    seen_dirs = set()
    for directory in candidate_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        normalized = os.path.normcase(os.path.abspath(directory))
        if normalized in seen_dirs:
            continue
        seen_dirs.add(normalized)
        search_dirs.append(directory)

    binaries = []
    collected = []
    for dll_name in sorted(MSVC_RUNTIME_DLLS):
        for directory in search_dirs:
            dll_path = os.path.join(directory, dll_name)
            if os.path.isfile(dll_path):
                binaries.append((dll_path, '.'))
                collected.append(dll_name)
                break
    if collected:
        print(
            "PyInstaller spec: bundled MSVC runtime DLLs:",
            ", ".join(sorted(set(collected))),
        )
    return binaries

def _to_binary_toc_entries(binaries):
    toc_entries = []
    for src_path, _ in binaries:
        toc_entries.append((os.path.basename(src_path), src_path, 'BINARY'))
    return toc_entries

def _strip_msvc_runtime_binaries(binaries):
    kept = []
    removed = []
    for entry in binaries:
        names = _entry_dll_names(entry)
        if any(name in MSVC_RUNTIME_DLLS for name in names):
            removed.extend(name for name in names if name in MSVC_RUNTIME_DLLS)
            continue
        kept.append(entry)
    if removed:
        print(
            "PyInstaller spec: excluded bundled MSVC runtime DLLs:",
            ", ".join(sorted(set(removed))),
        )
    return kept

# Qt6Core.dll links dynamically against the ICU that Windows itself ships in
# System32 (the unversioned names icuuc.dll / icuin.dll / icu.dll).
# Conda environments ship a *different* ICU build under those very same names
# in <env>/Library/bin, and that directory sits on PATH during a build, so
# PyInstaller happily bundles it into the app root where it shadows the system
# DLL.  Qt then dies at startup with:
#   ImportError: DLL load failed while importing QtCore:
#   The specified procedure could not be found.   (ERROR_PROC_NOT_FOUND)
# Windows' own ICU must win, so never bundle these names.
SYSTEM_ICU_DLL_NAMES = {'icu.dll', 'icuuc.dll', 'icuin.dll'}
_SYSTEM_ICU_DATA_RE = re.compile(r'^icudt\d*\.dll$')


def _strip_system_icu_binaries(binaries):
    kept = []
    removed = []
    for entry in binaries:
        shadowing = [
            name
            for name in _entry_dll_names(entry)
            if name in SYSTEM_ICU_DLL_NAMES or _SYSTEM_ICU_DATA_RE.match(name)
        ]
        if shadowing:
            removed.extend(shadowing)
            continue
        kept.append(entry)
    if removed:
        print(
            "PyInstaller spec: excluded system-shadowing ICU DLLs "
            "(Windows provides these in System32):",
            ", ".join(sorted(set(removed))),
        )
    return kept

onnxruntime_binaries = _collect_onnxruntime_dlls()
msvc_runtime_binaries = _collect_msvc_runtime_dlls()
matplotlib_datas = collect_data_files('matplotlib')

# torchvision loads its compiled ops extension at runtime with
# torch.ops.load_library(_get_extension_path("_C_stable")), i.e. a plain file
# path rather than an `import`, so PyInstaller's static analysis cannot see it
# and does not bundle torchvision/_C_stable.pyd or image_stable.pyd.  The
# loader swallows that failure silently (extension._load_library returns
# False), so the packaged app starts fine but later dies inside transformers
# with:  RuntimeError: operator torchvision::nms does not exist
# collect_dynamic_libs defaults to *.dll only, hence the explicit patterns
# (*.pyd is the Windows extension-module suffix).  The sibling jpeg/png/webp/
# zlib DLLs are the codec dependencies of image_stable.pyd.
torchvision_binaries = collect_dynamic_libs(
    'torchvision', search_patterns=['*.pyd', '*.dll'])

a = Analysis(
    [_p('anylabeling', 'app.py')],
    pathex=[_p('anylabeling')],
    binaries=onnxruntime_binaries + torchvision_binaries,
    datas=[
        (_p('anylabeling', 'configs', '*'), 'anylabeling/configs'),
        (_p('anylabeling', 'resources', '*'), 'anylabeling/resources'),
        (_p('anylabeling', 'views', '*'), 'anylabeling/views'),
        (_p('anylabeling', 'services', '*'), 'anylabeling/services'),
        (_p('anylabeling', 'services', 'auto_labeling', 'models'), 'anylabeling/services/auto_labeling/models'),
        (_p('assets', '*'), 'assets'),
        (_p('anylabeling', 'views', 'labeling', 'widgets', 'auto_labeling', 'auto_labeling.ui'), 'anylabeling/views/labeling/widgets/auto_labeling')
    ] + matplotlib_datas,
    hiddenimports=[
        'matplotlib',
        'matplotlib.backends.backend_agg',
        'matplotlib.font_manager',
        'matplotlib.mathtext',
        # Energy-spectrum pipeline: pandas/openpyxl are imported lazily inside
        # functions (Excel range-energy tables), so list them explicitly.
        'pandas',
        'openpyxl',
    ],
    hookspath=[],
    runtime_hooks=[_p('packaging', 'pyinstaller', 'runtime_hooks', 'ort_dll_bootstrap.py')],
    excludes=[],
)
a.binaries = _strip_msvc_runtime_binaries(a.binaries)
a.binaries = _strip_system_icu_binaries(a.binaries)
if msvc_runtime_binaries:
    a.binaries += _to_binary_toc_entries(msvc_runtime_binaries)
pyz = PYZ(a.pure, a.zipped_data)

# onedir (folder) vs onefile (single .exe) build:
#   - onedir (default) unpacks everything into a folder next to the exe at
#     build time, so there is no runtime unpacking and the
#     "failed to extract entry: LIBBZ2.dll" error cannot occur. It also
#     starts faster. This is the recommended mode for distribution.
#   - onefile embeds everything in one .exe and unpacks it to %TEMP% on every
#     launch; that runtime unpacking is exactly where the error above occurs.
#     Set X_ANYLABELING_ONEFILE=1 to build a single .exe instead.
build_onefile = os.environ.get('X_ANYLABELING_ONEFILE', '0') == '1'

exe_kwargs = dict(
    name=f'X-AnyLabeling-v{__version__}-CPU',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon=_p('anylabeling', 'resources', 'images', 'icon.icns'),
)

if build_onefile:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        **exe_kwargs,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        **exe_kwargs,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name='X-AnyLabeling-CPU',
        strip=False,
        upx=False,
    )

app = BUNDLE(
    exe,
    name='X-AnyLabeling.app',
    icon=_p('anylabeling', 'resources', 'images', 'icon.icns'),
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
