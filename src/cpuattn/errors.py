class UnsupportedError(ValueError):
    """A valid definition has no complete lowering on the selected backend."""


__all__ = ["UnsupportedError"]
