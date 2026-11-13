"""Prompt canonicalization shared by Pi training-compatible inference paths."""

from __future__ import annotations

import re


_ACTION_PREFIX = re.compile(r"^\s*<act:(?:joint|xyz)>\s*", re.IGNORECASE)


def canonicalize_molmospaces_prompt(prompt: object) -> str:
    """Remove the Octo-only action token from a MolmoSpaces Pi prompt.

    The shared simulator client prefixes task language with ``<act:joint>`` or
    ``<act:xyz>`` because Octo was trained with those tokens. MolmoSpaces RLDS
    prompts consumed by Pi are the bare authoritative task (for example,
    ``open the drawer``). Keeping the prefix at Pi inference creates a prompt
    distribution shift, so the Pi server strips only this known leading token.
    """

    if isinstance(prompt, bytes | bytearray):
        prompt = prompt.decode()
    return _ACTION_PREFIX.sub("", str(prompt), count=1).strip()
