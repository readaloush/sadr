import os, json, shutil, numpy as np, torch, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from common import *
D = '../docs'; os.makedirs(D + '/models', exist_ok=True); os.makedirs(D + '/samples', exist_ok=True)
# masked model (trained with the lungs hidden)
m = resnet18(False); m.load_state_dict(torch.load('../results/teacher_masked.pt')); m.eval()
fp = '../results/onnx/edges.onnx'
torch.onnx.export(m, batch(test[:1], mask=True)[0], fp, input_names=['image'], output_names=['logits'], dynamic_axes={'image': {0: 'n'}, 'logits': {0: 'n'}}, opset_version=17, dynamo=False)
ci = np.random.default_rng(1).choice(train, 256, replace=False)
class Rd(CalibrationDataReader):
    def __init__(s): s.it = iter(range(0, 256, 16))
    def get_next(s):
        i = next(s.it, None); return None if i is None else {'image': batch(ci[i:i+16], mask=True)[0].numpy()}
quant_pre_process(fp, fp + '.pre'); quantize_static(fp + '.pre', '../results/onnx/edges.int8.onnx', Rd(), quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8); os.remove(fp + '.pre')
for src, dst in [('teacher.int8.onnx', 'full.int8.onnx'), ('pruned70_ce.int8.onnx', 'small.int8.onnx'), ('edges.int8.onnx', 'edges.int8.onnx')]:
    shutil.copy('../results/onnx/' + src, D + '/models/' + dst)
# samples: Kermany test images only (CC BY 4.0), both classes; pick ones the full model gets right and wrong
R = json.load(open('../data/index.json'))
s = ort.InferenceSession(D + '/models/full.int8.onnx')
picks = []
for k in [1, 2]:
    idx = [i for i in test if Y[i] == k]
    rng = np.random.default_rng(k); rng.shuffle(idx)
    for i in idx[:4]:
        p = s.run(None, {'image': batch(np.array([i]))[0].numpy()})[0][0]; p = np.exp(p - p.max()); p /= p.sum()
        im = Image.open('../data/raw/' + R[i]['path']).convert('L'); im.thumbnail((640, 640))
        name = f'{CL[k].lower()}_{len(picks)}.jpg'; im.save(D + '/samples/' + name, quality=88)
        picks.append({'file': name, 'label': CL[k], 'source': R[i]['path'], 'ref_full': [round(float(v), 4) for v in p]})
json.dump(picks, open(D + '/samples/samples.json', 'w'), indent=1)
print([(p['file'], p['label'], CL[int(np.argmax(p['ref_full']))]) for p in picks])
print({f: round(os.path.getsize(D + '/models/' + f) / 2**20, 2) for f in os.listdir(D + '/models')})
