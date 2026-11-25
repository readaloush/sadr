"""Export to ONNX, int8 static quantization (QDQ, per-channel, 256 calibration images), accuracy through ONNX Runtime, and single-thread CPU latency."""
import os, time, glob, json, numpy as np, torch, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from common import *
OUT = '../results/onnx'; os.makedirs(OUT, exist_ok=True)
models = {'teacher': None}
for f in sorted(glob.glob('../results/pruned*_*.pth')): models[os.path.basename(f)[:-4]] = f
calib_idx = np.random.default_rng(1).choice(train, 256, replace=False)
class Reader(CalibrationDataReader):
    def __init__(s): s.it = iter(range(0, 256, 16))
    def get_next(s):
        i = next(s.it, None)
        return None if i is None else {'image': batch(calib_idx[i:i+16])[0].numpy()}
def run(sess, idx, bs=64):
    return np.concatenate([sess.run(None, {'image': batch(idx[i:i+bs])[0].numpy()})[0] for i in range(0, len(idx), bs)])
def latency(path, n=200):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider']); x = batch(test[:1])[0].numpy()
    for _ in range(20): s.run(None, {'image': x})
    t = []
    for _ in range(n): t0 = time.perf_counter(); s.run(None, {'image': x}); t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)
res = {}
for name, f in models.items():
    if f is None: m = resnet18(False); m.load_state_dict(torch.load('../results/teacher_full.pt'))
    else: m = torch.load(f, weights_only=False)
    m.eval(); fp = f'{OUT}/{name}.onnx'
    torch.onnx.export(m, batch(test[:1])[0], fp, input_names=['image'], output_names=['logits'], dynamic_axes={'image': {0: 'n'}, 'logits': {0: 'n'}}, opset_version=17, dynamo=False)
    pre = f'{OUT}/{name}.pre.onnx'; quant_pre_process(fp, pre)
    q = f'{OUT}/{name}.int8.onnx'
    quantize_static(pre, q, Reader(), quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    os.remove(pre)
    for kind, path in [('fp32', fp), ('int8', q)]:
        s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        P = run(s, test); P = np.exp(P - P.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        r = metrics(test, P); r['mb'] = os.path.getsize(path) / 2**20; r['ms'] = latency(path)
        res[f'{name}_{kind}'] = r
        print(f'{name:16s} {kind}: acc {r["acc"]:.4f} N-vs-P {r["normal_vs_pneumonia_acc"]:.4f} size {r["mb"]:.1f} MB latency {r["ms"]:.1f} ms', flush=True)
json.dump(res, open('../results/quant_bench.json', 'w'), indent=1)
