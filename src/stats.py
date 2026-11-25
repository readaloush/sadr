import json, numpy as np, onnxruntime as ort, glob, os
from common import *
rng = np.random.default_rng(0); y = Y[test]
def probs(path):
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    P = np.concatenate([s.run(None, {'image': batch(test[i:i+64])[0].numpy()})[0] for i in range(0, len(test), 64)]); return P.argmax(1)
names = ['teacher_fp32', 'teacher_int8', 'pruned30_ce_fp32', 'pruned30_kd_fp32', 'pruned50_ce_fp32', 'pruned50_kd_fp32', 'pruned70_ce_fp32', 'pruned70_kd_fp32', 'pruned70_ce_int8', 'pruned50_ce_int8']
pred = {n: probs('../results/onnx/' + n.replace('_fp32', '.onnx').replace('_int8', '.int8.onnx')) for n in names}
B = rng.integers(0, len(y), (2000, len(y))); out = {}
base = (pred['teacher_fp32'] == y)
for n in names:
    c = (pred[n] == y); acc = c[B].mean(1); d = (c[B].mean(1) - base[B].mean(1))
    out[n] = {'acc': float(c.mean()), 'ci95': [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
              'diff_vs_teacher': float(c.mean() - base.mean()), 'diff_ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    print(f"{n:18s} acc {100*c.mean():.1f} [{100*out[n]['ci95'][0]:.1f}, {100*out[n]['ci95'][1]:.1f}]  vs teacher {100*out[n]['diff_vs_teacher']:+.1f} [{100*out[n]['diff_ci95'][0]:+.1f}, {100*out[n]['diff_ci95'][1]:+.1f}]")
# full teacher, applied to test images with the lungs blacked out (no retraining)
m = resnet18(False); m.load_state_dict(torch.load('../results/teacher_full.pt'))
r = metrics(test, predict(m, test, mask=True)); out['teacher_on_masked_test'] = r
print('teacher on masked test: acc', round(r['acc'], 4), r['cm'])
json.dump(out, open('../results/stats.json', 'w'), indent=1)
