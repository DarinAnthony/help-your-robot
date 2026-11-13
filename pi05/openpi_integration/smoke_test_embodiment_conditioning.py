"""One-batch smoke test for the installed openpi RLDS integration.

Run with the openpi environment, for example:
  cd /path/to/openpi
  .venv/bin/python /path/to/repo/openpi_integration/smoke_test_embodiment_conditioning.py \
      /path/to/rlds_root
"""

from __future__ import annotations

import argparse

import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.molmospaces_rlds_dataset import MolmoSpacesRldsDataset
from openpi.training.molmospaces_rlds_dataset import RLDSDataset


def first_prompt(data_dir: str, mode: str) -> tuple[str, tuple[int, ...]]:
    dataset = MolmoSpacesRldsDataset(
        data_dir,
        1,
        (RLDSDataset("molmospaces_emb_mix", "1.0.0"),),
        shuffle=False,
        action_chunk_size=2,
        num_parallel_reads=1,
        num_parallel_calls=1,
        embodiment_conditioning=mode,
    )
    batch = next(iter(dataset))
    return batch["prompt"][0].decode(), batch["actions"].shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    legacy_prompt, legacy_shape = first_prompt(args.data_dir, "none")
    text_prompt, text_shape = first_prompt(args.data_dir, "text")

    assert legacy_prompt in text_prompt
    assert "; embodiment: robot " in text_prompt
    assert legacy_shape == text_shape == (1, 2, 10)
    _, token_mask = PaligemmaTokenizer(max_len=200).tokenize(text_prompt, np.zeros(10, np.float32))
    token_count = int(token_mask.sum())
    assert token_count < 200, "conditioned prompt truncated the pi0.5 State/Action suffix"
    print(f"legacy prompt: {legacy_prompt}")
    print(f"conditioned prompt: {text_prompt}")
    print(f"action shape: {text_shape}")
    print(f"pi0.5 prefix tokens: {token_count}/200")


if __name__ == "__main__":
    main()
