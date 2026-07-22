"""Compatibility shim for Python builds without stdlib ``_lzma``.

SGLang starts with ``scripts/sglang_stubs`` at the front of ``PYTHONPATH``.
The real virtualenv has ``backports.lzma`` installed, so expose that full
implementation here instead of a minimal stub. Libraries such as joblib inspect
``LZMAFile`` for file-object methods at import time.
"""

try:
    from backports.lzma import *  # noqa: F401,F403
except Exception as exc:  # pragma: no cover - only used on broken envs
    FORMAT_AUTO = 0
    FORMAT_XZ = 1
    FORMAT_ALONE = 2
    FORMAT_RAW = 3
    CHECK_NONE = 0
    CHECK_CRC32 = 1
    CHECK_CRC64 = 4
    CHECK_SHA256 = 10
    CHECK_ID_MAX = 15
    PRESET_DEFAULT = 6
    PRESET_EXTREME = 1 << 31
    FILTER_LZMA1 = 0x4000000000000001
    FILTER_LZMA2 = 0x21
    FILTER_DELTA = 0x03
    FILTER_X86 = 0x04
    FILTER_IA64 = 0x05
    FILTER_ARM = 0x07
    FILTER_ARMTHUMB = 0x08
    FILTER_POWERPC = 0x05
    FILTER_SPARC = 0x09
    MF_HC3 = 0x03
    MF_HC4 = 0x04
    MF_BT2 = 0x12
    MF_BT3 = 0x13
    MF_BT4 = 0x14
    MODE_FAST = 1
    MODE_NORMAL = 2

    class LZMAError(Exception):
        pass

    class LZMAFile:
        def __init__(self, *args, **kwargs):
            raise LZMAError(f"lzma support is unavailable: {exc}")

        def read(self, *args, **kwargs):
            raise LZMAError(f"lzma support is unavailable: {exc}")

        def write(self, *args, **kwargs):
            raise LZMAError(f"lzma support is unavailable: {exc}")

        def seek(self, *args, **kwargs):
            raise LZMAError(f"lzma support is unavailable: {exc}")

        def tell(self, *args, **kwargs):
            raise LZMAError(f"lzma support is unavailable: {exc}")

    def open(*args, **kwargs):
        raise LZMAError(f"lzma support is unavailable: {exc}")

    def compress(*args, **kwargs):
        raise LZMAError(f"lzma support is unavailable: {exc}")

    def decompress(*args, **kwargs):
        raise LZMAError(f"lzma support is unavailable: {exc}")

    class LZMACompressor:
        def __init__(self, *args, **kwargs):
            raise LZMAError(f"lzma support is unavailable: {exc}")

    class LZMADecompressor:
        def __init__(self, *args, **kwargs):
            raise LZMAError(f"lzma support is unavailable: {exc}")

    def is_check_supported(check):
        return False
