"""CPU architecture, numerical, gradient, optimizer and checkpoint checks."""
import copy
import io
import json
import torch
import yaml
from torch.nn import functional as F
from models.ald import DynConvLayer, FANet
from models.yolo import Model
from utils.ald_wiou import WiseIoU
from utils.loss_tal_dual import ComputeLoss
from utils.torch_utils import smart_optimizer


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    layer = DynConvLayer(3, 8).eval()
    x = torch.randn(2, 3, 16, 16)
    # Controlled routing parameters avoid a random all-inactive ReLU fixture.
    with torch.no_grad():
        layer.router[1].weight.fill_(0.2)
        layer.router[1].bias.fill_(0.5)
        layer.router[3].weight.copy_(torch.arange(4.).view(4, 1, 1, 1).expand_as(layer.router[3].weight))
        layer.router[3].bias.zero_()
        x[1].add_(1.0)
    alpha = layer.routing_weights(x)
    ref = sum(F.conv2d(x, layer.weight[k], padding=1) * alpha[:, k, None, None, None]
              for k in range(layer.experts))
    torch.testing.assert_close(layer(x), layer.act(layer.bn(ref)), atol=2e-6, rtol=2e-5)
    assert not torch.allclose(alpha[0], alpha[1]), 'Router must depend on input'

    for mode in ('wiou_v1', 'wiou_v3'):
        loss = WiseIoU(mode)
        identical = torch.tensor([[0., 0., 2., 2.]])
        value, iou = loss(identical, identical)
        torch.testing.assert_close(value, torch.zeros_like(value))
        p = torch.tensor([[0., 0., 1., 1.], [3., 3., 4., 4.]], requires_grad=True)
        v, _ = loss(p, identical.repeat(2, 1))
        v.sum().backward()
        assert torch.isfinite(v).all() and torch.isfinite(p.grad).all()
        before = loss.iou_mean.clone()
        with torch.no_grad(): loss(p, identical.repeat(2, 1))
        torch.testing.assert_close(before, loss.iou_mean)
        restored = WiseIoU(mode)
        restored.load_state_dict(loss.state_dict())
        torch.testing.assert_close(restored.iou_mean, loss.iou_mean)

    model = Model('models/detect/ald-yolov9c.yaml', ch=3, nc=15).cpu()
    assert sum(isinstance(m, DynConvLayer) for m in model.modules()) == 2
    assert sum(isinstance(m, FANet) for m in model.modules()) == 3
    assert model.model[-1].f == [31, 34, 37, 38, 39, 40]
    with open('data/hyps/hyp.ald.yaml') as f: model.hyp = yaml.safe_load(f)
    criterion = ComputeLoss(model)
    optimizer = smart_optimizer(model, 'SGD', .001, .931, .0005)
    registered = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(registered) == len(set(registered))
    # Upstream also registers frozen DFL projection weights; all trainable
    # parameters must be included exactly once, with no foreign parameters.
    assert {id(p) for p in model.parameters() if p.requires_grad} <= set(registered)
    assert set(registered) <= {id(p) for p in model.parameters()}
    model.train()
    pred = model(torch.rand(2, 3, 64, 64))
    assert len(pred) == 2 and all(len(branch) == 3 for branch in pred)
    targets = torch.tensor([[0, 0, .5, .5, .4, .4], [1, 14, .4, .4, .3, .3]])
    value, parts = criterion(pred, targets)
    assert torch.isfinite(value)
    value.backward()
    for module in model.modules():
        if isinstance(module, (DynConvLayer, FANet)):
            for name, param in module.named_parameters():
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
    optimizer.step()
    optimizer.zero_grad()
    empty_value, _ = criterion(model(torch.rand(2, 3, 64, 64)), torch.empty(0, 6))
    assert torch.isfinite(empty_value)
    model.eval()
    with torch.no_grad():
        result = model(torch.rand(1, 3, 64, 64))
    assert len(result[0]) == 2 and result[0][1].shape[1] == 19
    # Round-trip model serialization uses the same classes needed by val/detect.
    buffer = io.BytesIO()
    torch.save({'model': model, 'ald_loss_state': [criterion.bbox_loss.state_dict(),
                                                 criterion.bbox_loss2.state_dict()]}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)
    restored = ComputeLoss(checkpoint['model'])
    restored.bbox_loss.load_state_dict(checkpoint['ald_loss_state'][0])
    restored.bbox_loss2.load_state_dict(checkpoint['ald_loss_state'][1])
    print(json.dumps({'status': 'ALD CHECK PASSED', 'dynamic_layers': 2,
                      'FA_Net_modules': 3, 'strides': model.stride.tolist(),
                      'parameters_training_graph': sum(p.numel() for p in model.parameters()),
                      'synthetic_loss': float(value.detach()),
                      'note': 'CPU synthetic checks, not dataset accuracy or GPU validation'}, indent=2))

if __name__ == '__main__':
    main()
