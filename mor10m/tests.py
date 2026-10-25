"""Causality, parameter sharing, router gradients, and checkpoint round-trip."""
import json
from pathlib import Path
import tempfile
import torch
from safetensors.torch import save_file, load_file
from model import MoR, Config, parameter_count


def run():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    c=Config(vocab_size=128,d_model=32,d_ff=64,n_q_heads=2,n_kv_heads=1)
    m=MoR(c).eval()
    x=torch.randint(0,c.vocab_size,(3,24))
    y=x.clone(); y[:,12:]=torch.randint(0,c.vocab_size,(3,12))
    with torch.no_grad():
        a=m(x); b=m(y)
        torch.testing.assert_close(a[:,:12],b[:,:12],atol=2e-6,rtol=2e-5)
        for depth in (0,1,2):
            forced=torch.tensor(depth)
            torch.testing.assert_close(m(x,forced_depth=forced)[:,:12],m(y,forced_depth=forced)[:,:12],atol=2e-6,rtol=2e-5)
        # Sparse passes with completely inactive rows must remain finite.
        forced=torch.tensor([[0],[2],[1]])
        assert torch.isfinite(m(x,forced_depth=forced)).all()
    m.train(); out,aux,stats=m(x,return_aux=True)
    loss=torch.nn.functional.cross_entropy(out.reshape(-1,128),x.roll(-1,1).reshape(-1))+aux
    loss.backward()
    assert m.router.weight.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    clone=MoR(c); clone.load_state_dict(m.state_dict()); clone.eval(); m.eval()
    torch.testing.assert_close(clone(x),m(x))
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'model.safetensors'
        save_file(m.state_dict(),str(path)); clone.load_state_dict(load_file(str(path)))
        torch.testing.assert_close(clone(x),m(x))
    # A fixed tiny batch must be learnable, not just produce finite gradients.
    opt=torch.optim.AdamW(m.parameters(),lr=.003)
    first=float(loss.detach())
    for _ in range(35):
        opt.zero_grad(); out,aux,_=m(x,return_aux=True)
        loss=torch.nn.functional.cross_entropy(out.reshape(-1,128),x.roll(-1,1).reshape(-1))+aux
        loss.backward(); opt.step()
    assert float(loss.detach()) < first*.75,(first,float(loss.detach()))
    # Optimizer moments must survive a real on-disk round trip at FP32.
    with tempfile.TemporaryDirectory() as directory:
        from train import checkpoint
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        cp=checkpoint(Path(directory),m,opt,scaler,35,2240,2275,{'loss':float(loss.detach())})
        saved=torch.load(cp/'training_state.pt',map_location='cpu',weights_only=False)
        for entry in saved['optimizer']['state'].values():
            assert entry['exp_avg'].dtype==torch.float32
            assert entry['exp_avg_sq'].dtype==torch.float32
        restored=torch.optim.AdamW(clone.parameters(),lr=.003)
        restored.load_state_dict(saved['optimizer'])
        assert saved['cursor']==2275
    count=parameter_count()
    assert 9_900_000 <= count <= 10_100_000,count
    print(json.dumps({"passed":True,"parameters":count,"causal":True,"router_gradients":True,"initial_loss":first,"overfit_loss":float(loss.detach())}))

if __name__=="__main__": run()
