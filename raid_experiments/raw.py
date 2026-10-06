"""Disabled legacy raw-image entry point.

The active raw-image encoder is standalone MAE in :mod:`src.mae_encoder` and
is called only by the versioned few-shot study. The former pilot loader is not
available as an execution path.
"""


def load_raw_encoder_pair(*args, **kwargs):
    raise RuntimeError("legacy raw-image runner is disabled; use the v3 standalone-MAE study")


def encode_trainable(*args, **kwargs):
    raise RuntimeError("legacy raw-image runner is disabled; use the v3 standalone-MAE study")


def assert_lora_isolated(*args, **kwargs):
    raise RuntimeError("legacy raw-image runner is disabled; use the v3 standalone-MAE study")
