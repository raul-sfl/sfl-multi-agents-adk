from typing import Callable, List


def build_gemini_tools(funcs: List[Callable]) -> List[Callable]:
    """Return Python callables directly.

    google-generativeai >= 0.5 accepts Python functions as tools and builds
    the JSON schema automatically from type hints and docstrings.
    This avoids manual protos.Schema construction which is fragile across SDK versions.
    """
    return list(funcs)
