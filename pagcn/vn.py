"""PA-GCN (NAK) + heterogeneous virtual node, after MPNN+VN_G (Southern et al., ICLR 2025).
Per layer l: local  KH = K H^(l-1)                                  (NAK, as in the paper)
             global g = sum_i s_i psi(h_i) / sum_i s_i,  s_i = sigmoid(w^T h_i + c)   (node-dependent weights)
             Z = 0.9 KH + 0.1 H^(0) + lambda_l * 1 g^T,  lambda_l learnable, initialized at 0
then the GCNII update as before. Cost O(nd) per layer; no handcrafted input; the graph is not changed."""
import math, torch
from . import model as PF
from .model import D


class VNGCN(PF.PAGCN):
    def __init__(self, G, cfg, seed):
        super().__init__(G, cfg, seed)
        Lc = len(self.Ws); torch.manual_seed(seed + 3000)
        self.psi = torch.nn.ModuleList([torch.nn.Linear(D, D) for _ in range(Lc)])
        self.gate = torch.nn.ModuleList([torch.nn.Linear(D, 1) for _ in range(Lc)])
        self.lam = torch.nn.Parameter(torch.zeros(Lc))

    def vn(self, l, H):                              # H: (..., n, D) -> (..., 1, D)
        s = torch.sigmoid(self.gate[l](H)); v = torch.relu(self.psi[l](H))
        return (s * v).sum(-2, keepdim=True) / (s.sum(-2, keepdim=True) + 1e-8)

    def layer(self, l, H, H0, al):                   # H, H0: (n, D) or (B, n, D)
        if H.dim() == 2:
            KH = self.prop(al, H)
        else:
            Bn, n, _ = H.shape
            KH = self.prop(al, H.permute(1, 0, 2).reshape(n, Bn * D)).reshape(n, Bn, D).permute(1, 0, 2)
        Z = 0.9 * KH + 0.1 * H0 + self.lam[l] * self.vn(l, H)
        bl = math.log(0.5 / (l + 1) + 1)
        return torch.relu((1 - bl) * Z + bl * self.Ws[l](Z))

    def forward(self, states, want_energy=False, return_hidden=False):
        G = self.G; Bn = len(states); al = self.alpha()
        X = torch.cat([self.X.unsqueeze(0).expand(Bn, -1, -1), self.state_feats(states)], -1)
        H0 = self.inp(X); H = H0; Hs = [H0]
        for l in range(len(self.Ws)):
            H = self.layer(l, H, H0, al); Hs.append(H)
        s = self.head(torch.cat(Hs, -1)).squeeze(-1)
        if return_hidden: return s, Hs
        if not want_energy: return s
        return s, self.energies(Hs, al)


def train(G, obj, cfg, seed, kmax, log=None):
    orig = PF.PAGCN; PF.PAGCN = VNGCN
    try: return PF.train(G, obj, cfg, seed, kmax, log)
    finally: PF.PAGCN = orig
