"""Same softmax one-vs-rest Dice/Focal formula, evaluated in logit space without hard probability clamps."""
import torch
from torch.nn import functional as F
from losses import PaperLoss


class StablePaperLoss(PaperLoss):
    def forward(self,logits,target):
        z=logits.float();c=z.shape[1]
        if c<2:raise ValueError('Multiclass softmax needs at least two classes')
        logp=F.log_softmax(z,dim=1);probs=logp.exp()
        labels=F.one_hot(target,c).permute(0,3,1,2).float()
        dims=(0,2,3)
        dice=1-((2*(probs*labels).sum(dims)+self.smooth)/(probs.sum(dims)+labels.sum(dims)+self.smooth)).mean()
        # log(1-p_c) = logsumexp(z_except_c) - logsumexp(z_all).
        # Unlike log1p(-softmax), this remains finite when p_c rounds to one.
        denominator=torch.logsumexp(z,dim=1)
        lognot=torch.stack([torch.logsumexp(torch.cat((z[:,:i],z[:,i+1:]),dim=1),dim=1)-denominator for i in range(c)],dim=1)
        focal=-(self.alpha*(1-probs).pow(self.gamma)*labels*logp+
                (1-self.alpha)*probs.pow(self.gamma)*(1-labels)*lognot).mean()
        return self.dice_weight*dice+self.focal_weight*focal
