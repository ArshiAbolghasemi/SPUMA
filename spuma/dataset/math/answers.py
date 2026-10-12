"""MATH answer extraction (\\boxed{})."""


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}), with nested braces; None if absent/unbalanced."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    i = text.index("{", start) + 1
    depth = 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j].strip()
    return None
