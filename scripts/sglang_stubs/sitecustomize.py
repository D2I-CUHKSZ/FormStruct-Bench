"""Runtime compatibility fixes for the local SGLang environment.

This directory is prepended to ``PYTHONPATH`` only for SGLang subprocesses.
Keep these patches narrow so they do not affect the vLLM virtualenv or normal
repository tooling.
"""

from __future__ import annotations

import inspect


def _patch_transformers_tokenizers() -> None:
    try:
        from transformers.tokenization_utils import PreTrainedTokenizer
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    except Exception:
        return

    for cls in (PreTrainedTokenizer, PreTrainedTokenizerFast):
        original = getattr(cls, "_batch_encode_plus", None)
        if original is None or getattr(original, "_formtsr_drops_video_metadata", False):
            continue
        signature = inspect.signature(original)
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        accepted_kwargs = set(signature.parameters)

        def wrapped(
            self,
            *args,
            __original=original,
            __accepts_var_kwargs=accepts_var_kwargs,
            __accepted_kwargs=accepted_kwargs,
            **kwargs,
        ):
            if not __accepts_var_kwargs:
                kwargs = {key: value for key, value in kwargs.items() if key in __accepted_kwargs}
            return __original(self, *args, **kwargs)

        wrapped._formtsr_drops_video_metadata = True  # type: ignore[attr-defined]
        setattr(cls, "_batch_encode_plus", wrapped)


_patch_transformers_tokenizers()
