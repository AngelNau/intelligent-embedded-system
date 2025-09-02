from typing import Any
import torchvision.models as m

def transfer_and_slice(pretrained_state: dict[str, Any], target_model: m.MobileNetV2):
    """
    Copy tensors from pretrained_state (a state_dict) into target_model.state_dict().
    Where shapes differ and slicing is possible, copy an initial slice.
    Returns (n_copied, n_sliced, n_skipped).
    """
    target_state = target_model.state_dict()
    new_state = {}
    n_copied = n_sliced = n_skipped = 0

    for k, tgt_tensor in target_state.items():
        if k not in pretrained_state:
            # no pretrained info for this key
            new_state[k] = tgt_tensor
            n_skipped += 1
            continue

        src_tensor = pretrained_state[k]
        if src_tensor.shape == tgt_tensor.shape:
            new_state[k] = src_tensor.clone()
            n_copied += 1
            continue

        try:
            if src_tensor.ndim != tgt_tensor.ndim:
                new_state[k] = tgt_tensor
                n_skipped += 1
                continue

            slices = tuple(slice(0, min(s, t)) for s, t in zip(src_tensor.shape, tgt_tensor.shape))
            filled = tgt_tensor.clone()
            filled[slices] = src_tensor[slices]
            new_state[k] = filled
            n_sliced += 1
        except Exception:
            new_state[k] = tgt_tensor
            n_skipped += 1

    target_model.load_state_dict(new_state)
    return n_copied, n_sliced, n_skipped
