import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, torchvision, time, json
torch.set_num_threads(2)
D = np.load('../data/img160.npz'); X, Y, SP = D['X'], D['y'], D['split']
CL = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
rng = np.random.default_rng(0)
trainval = np.where(SP != 'test')[0]; test = np.where(SP == 'test')[0]
# our own validation split: 10% of train+val, stratified
val = np.concatenate([rng.choice(trainval[Y[trainval] == k], size=int(0.1 * (Y[trainval] == k).sum()), replace=False) for k in range(4)])
train = np.setdiff1d(trainval, val)
MEAN, STD = 0.449, 0.226

def batch(idx, aug=False, mask=False):
    x = torch.from_numpy(X[idx]).float().div(255)
    if aug:
        x = torch.stack([torchvision.transforms.functional.affine(t[None], angle=float(np.random.uniform(-8, 8)), translate=[int(np.random.randint(-8, 9)), int(np.random.randint(-8, 9))], scale=float(np.random.uniform(0.92, 1.08)), shear=[0.0])[0] for t in x])
        x = x * (1 + 0.15 * (torch.rand(len(idx), 1, 1) - 0.5)) + 0.1 * (torch.rand(len(idx), 1, 1) - 0.5)
    if mask: x[:, 24:136, 24:136] = 0          # hide the central 70% of the image (the lungs)
    x = (x - MEAN) / STD
    return x[:, None].expand(-1, 3, -1, -1).contiguous(), torch.from_numpy(Y[idx]).long()

def resnet18(pretrained=True):
    m = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None); m.fc = nn.Linear(512, 4); return m

def predict(m, idx, mask=False, bs=128):
    m.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, _ = batch(idx[i:i+bs], mask=mask); P.append(m(x).softmax(1).numpy())
    return np.concatenate(P)

def metrics(idx, P):
    y = Y[idx]; p = P.argmax(1); cm = np.zeros((4, 4), int)
    for a, b in zip(y, p): cm[a, b] += 1
    f1 = []
    for k in range(4):
        tp = cm[k, k]; se = tp / max(cm[k].sum(), 1); pp = tp / max(cm[:, k].sum(), 1); f1.append(2*se*pp/max(se+pp, 1e-9))
    np_mask = np.isin(y, [1, 2])           # Normal vs Pneumonia: the one pair that comes from the same source
    return {'acc': float((p == y).mean()), 'macro_f1': float(np.mean(f1)), 'f1': [float(v) for v in f1], 'cm': cm.tolist(),
            'normal_vs_pneumonia_acc': float((p[np_mask] == y[np_mask]).mean())}

def train_model(m, epochs, lr=1e-3, mask=False, teacher=None, T=4.0, alpha=0.7, bs=64, log=True, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-4)
    steps = epochs * (len(train) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.2)
    cnt = np.bincount(Y[train], minlength=4); w = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32); w /= w.mean()
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(train); t0 = time.time()
        for i in range(0, len(perm) - bs + 1, bs):
            x, y = batch(perm[i:i+bs], aug=True, mask=mask); out = m(x)
            loss = Fn.cross_entropy(out, y, weight=w)
            if teacher is not None:
                with torch.no_grad(): tl = teacher(x)
                kd = Fn.kl_div(Fn.log_softmax(out / T, 1), Fn.softmax(tl / T, 1), reduction='batchmean') * T * T
                loss = (1 - alpha) * loss + alpha * kd
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if log:
            mv = metrics(val, predict(m, val, mask=mask)); print(f'  ep{ep} {time.time()-t0:.0f}s val acc {mv["acc"]:.4f} N-vs-P {mv["normal_vs_pneumonia_acc"]:.4f}', flush=True)
    return m
