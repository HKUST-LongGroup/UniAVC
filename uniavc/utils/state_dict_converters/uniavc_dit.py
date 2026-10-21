"""State-dict converter for a UniAVC DiT checkpoint.

A released UniAVC export already carries the bare key names the model uses
(`attn1.q_norm.weight`, `transformer_blocks...`), but a checkpoint written mid-run by
the trainer still has `module.pipe.dit.` and similar wrappers. Peeling those here means
either file loads through the normal registry path with no special casing at the call
site.
"""

_PREFIXES = ("module.pipe.dit.", "module.dit.", "pipe.dit.", "dit.", "module.")


def UniAVCStateDictConverter(state_dict):
    converted = {}
    for key, value in state_dict.items():
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        converted[key] = value
    return converted
