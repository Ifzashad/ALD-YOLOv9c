"""Transfer upstream dual YOLOv9c initialization into each independent ablation."""
def adapt_state_dict(state, source, target):
    if len(source.model) != 39 or source.model[1].__class__.__name__ != 'Conv':
        return state
    if len(target.model) not in (39, 42):
        raise ValueError('Unsupported target architecture')
    head = len(target.model) - 1
    out = {(k.replace('model.38.', f'model.{head}.', 1) if k.startswith('model.38.') else k): v for k,v in state.items()}
    for idx in (1, 2):
        key = f'model.{idx}.conv.weight'
        if target.model[idx].__class__.__name__ == 'DynConvLayer' and key in out:
            out[f'model.{idx}.weight'] = out.pop(key).unsqueeze(0).repeat(target.model[idx].experts, 1, 1, 1, 1)
    return out
