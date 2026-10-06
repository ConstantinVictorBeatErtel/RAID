"""Removed legacy encoder entry point; kept only to fail closed."""
def __getattr__(name):
    raise RuntimeError("This legacy RAID encoder is permanently disabled; use src.mae_encoder.MAEImageEncoder")
