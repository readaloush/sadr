"""Exact and near-duplicate images between the test split and train/val (pHash candidates, NCC on 64x64 confirms)."""
import json, hashlib, numpy as np, collections
from PIL import Image
R = json.load(open('../data/index.json')); root = '../data/raw/'
H = np.array([[c == '1' for c in r['phash']] for r in R]); sp = np.array([r['split'] for r in R]); y = np.array([r['y'] for r in R])
md5 = [hashlib.md5(open(root + r['path'], 'rb').read()).hexdigest() for r in R]
def th(i):
    a = np.asarray(Image.open(root + R[i]['path']).convert('L').resize((64, 64)), np.float32).ravel(); a -= a.mean(); return a / (np.linalg.norm(a) + 1e-6)
te = np.where(sp == 'test')[0]; tr = np.where(sp != 'test')[0]
exact = sum(1 for i in te if md5[i] in {md5[j] for j in tr})
cache = {}
def T(i):
    if i not in cache: cache[i] = th(i)
    return cache[i]
best = {}
for i in te:
    d = (H[tr] != H[i]).sum(1); cand = tr[d <= 12]
    best[i] = max((float(T(i) @ T(j)), int(j)) for j in cand) if len(cand) else (0.0, -1)
CL = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
out = {'exact_test_in_train': exact}
for thr in [0.95, 0.98]:
    tw = [i for i in te if best[i][0] >= thr]
    out[f'twins_ncc_{thr}'] = {CL[k]: int(sum(1 for i in tw if y[i] == k)) for k in range(4)}
    out[f'twins_ncc_{thr}_label_differs'] = int(sum(1 for i in tw if y[best[i][1]] != y[i]))
out['test_per_class'] = {CL[k]: int((y[te] == k).sum()) for k in range(4)}
# duplicates inside the whole dataset (exact md5)
c = collections.Counter(md5); out['exact_duplicate_groups_all'] = sum(1 for v in c.values() if v > 1)
json.dump({'summary': out, 'twins': {R[i]['path']: [best[i][0], R[best[i][1]]['path'] if best[i][1] >= 0 else None] for i in te}}, open('../results/audit_dupes.json', 'w'))
print(json.dumps(out, indent=1))
