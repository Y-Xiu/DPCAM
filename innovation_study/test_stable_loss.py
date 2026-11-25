import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from losses import PaperLoss
from innovation_study.stable_loss import StablePaperLoss


def test_same_formula_and_gradients_away_from_saturation():
    torch.manual_seed(4);z=torch.randn(2,5,8,8,requires_grad=True);y=torch.randint(0,5,(2,8,8))
    old=PaperLoss()(z,y);new=StablePaperLoss()(z,y)
    torch.testing.assert_close(old,new,atol=1e-7,rtol=1e-6)
    a=torch.autograd.grad(old,z,retain_graph=True)[0];b=torch.autograd.grad(new,z)[0]
    torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-5)


def test_confident_wrong_prediction_keeps_corrective_gradient():
    z=torch.tensor([[[[1000.]], [[-1000.]], [[-1000.]], [[-1000.]], [[-1000.]]]],requires_grad=True)
    y=torch.ones(1,1,1,dtype=torch.long)
    old=PaperLoss()(z,y);new=StablePaperLoss()(z,y)
    a=torch.autograd.grad(old,z,retain_graph=True)[0];b=torch.autograd.grad(new,z)[0]
    assert torch.isfinite(old) and torch.isfinite(new) and torch.isfinite(b).all()
    assert a.abs().max()==0 and b[0,1,0,0]<0 and b[0,0,0,0]>0
    # Gradient descent increases the true-class logit and decreases wrong background.
    assert StablePaperLoss()(z.detach()-.1*b,y)<new.detach()
