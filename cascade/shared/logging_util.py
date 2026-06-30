"""Keep cascade's own loggers visible alongside bittensor.

bittensor's logging machine, on import, silences every *other* logger by setting
it to ``CRITICAL`` (it keeps only bittensor's own output). That swallows all
``cascade.*`` ``INFO``/``WARNING``/``ERROR`` messages from the trainer and
validator services, which makes them look hung when they are working fine.

:func:`restore_cascade_logging` imports bittensor (forcing that one-time silence
to happen now) and then puts the ``cascade.*`` loggers back at the requested
level. Call it once in a service ``main`` before entering the run loop.
"""

from __future__ import annotations

import logging


def restore_cascade_logging(level_name: str = "INFO") -> None:
    """Restore ``cascade.*`` logger levels after bittensor silences them."""
    # Force bittensor's logging machine to initialise now (it silences other
    # loggers on first import); doing it here means our restore below is the
    # last word. Best-effort: if bittensor isn't installed there's nothing to undo.
    try:
        import bittensor  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.getLogger("cascade").setLevel(level)
    for name in list(logging.root.manager.loggerDict):
        if name == "cascade" or name.startswith("cascade."):
            logging.getLogger(name).setLevel(level)
