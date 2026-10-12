"""CPU structural, transfer and forward/backward checks for each ablation."""
import gc
import torch,yaml
from models.yolo import Model
from models.ald import DynConvLayer,FANet
from utils.ald_transfer import adapt_state_dict
from utils.loss_tal_dual import ComputeLoss
from utils.general import intersect_dicts

def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    source=Model('models/detect/yolov9-c.yaml',ch=3,nc=80).cpu()
    for name,dyn,fa in [('baseline',0,0),('dyn-only',2,0),('fa-only',0,3),('wiou-only',0,0),('full',2,3)]:
        m=Model(f'models/ablations/{name}.yaml',ch=3,nc=15).cpu()
        assert sum(isinstance(x,DynConvLayer) for x in m.modules())==dyn
        assert sum(isinstance(x,FANet) for x in m.modules())==fa
        state=adapt_state_dict(source.state_dict(),source,m)
        compatible=intersect_dicts(state,m.state_dict())
        m.load_state_dict(compatible,strict=False)
        head=len(m.model)-1
        key=f'model.{head}.cv2.0.0.conv.weight'
        torch.testing.assert_close(m.state_dict()[key],source.state_dict()['model.38.cv2.0.0.conv.weight'])
        for i in (1,2):
            expected=source.model[i].conv.weight
            actual=m.model[i].weight[0] if dyn else m.model[i].conv.weight
            torch.testing.assert_close(actual,expected)
        m.hyp=yaml.safe_load(open(f'data/hyps/ablations/{name}.yaml'))
        criterion=ComputeLoss(m)
        assert (criterion.bbox_loss.wiou is not None)==(name in ('wiou-only','full'))
        m.train()
        loss,_=criterion(m(torch.rand(2,3,64,64)),torch.tensor([[0,0,.5,.5,.4,.4],[1,14,.4,.4,.3,.3]]))
        assert torch.isfinite(loss)
        loss.backward()
        for x in m.modules():
            if isinstance(x,(DynConvLayer,FANet)):
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in x.parameters())
        print('PASS',name,'parameters',sum(p.numel() for p in m.parameters()),'transferred',len(compatible),flush=True)
        del m,criterion,loss,state,compatible
        gc.collect()
    print('ALL ABLATION CHECKS PASSED (CPU synthetic; not training results)')

if __name__=='__main__':main()
