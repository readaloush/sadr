"""Class activation maps. ResNet-18 ends in global average pooling + one linear layer, so CAM_k = sum_c w_kc * A_c over the last
feature map (5x5 at 160 px) is exact: averaging CAM_k over space gives the logit minus the bias. Exported as a second ONNX output."""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from common import *
class WithCam(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        m = s.m; f = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))))
        logits = m.fc(f.mean((2, 3)))
        cam = torch.einsum('kc,nchw->nkhw', m.fc.weight, f)
        return logits, cam
out = {}
for name, wfile, mask in [('full', 'teacher_full.pt', False), ('edges', 'teacher_masked.pt', True)]:
    m = resnet18(False); m.load_state_dict(torch.load('../results/' + wfile)); m.eval(); w = WithCam(m).eval()
    # sanity: CAM mean + bias == logits
    x = batch(test[:4], mask=mask)[0]; l, c = w(x); assert torch.allclose(c.mean((2, 3)) + m.fc.bias, l, atol=1e-4)
    fp = f'../results/onnx/{name}_cam.onnx'
    torch.onnx.export(w, x[:1], fp, input_names=['image'], output_names=['logits', 'cam'], dynamic_axes={'image': {0: 'n'}, 'logits': {0: 'n'}, 'cam': {0: 'n'}}, opset_version=17, dynamo=False)
    ci = np.random.default_rng(1).choice(train, 256, replace=False)
    class Rd(CalibrationDataReader):
        def __init__(s): s.it = iter(range(0, 256, 16))
        def get_next(s):
            i = next(s.it, None); return None if i is None else {'image': batch(ci[i:i+16], mask=mask)[0].numpy()}
    q = f'../results/onnx/{name}_cam.int8.onnx'
    quant_pre_process(fp, fp + '.pre'); quantize_static(fp + '.pre', q, Rd(), quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8); os.remove(fp + '.pre')
    if name != 'full': continue
    # where does the full model look? share of positive CAM (predicted class) inside the centre box that the masked experiment hides
    s = ort.InferenceSession(q); share = []; pred = []
    for i in range(0, len(test), 64):
        lg, cm = s.run(None, {'image': batch(test[i:i+64])[0].numpy()})
        k = lg.argmax(1); cmk = torch.from_numpy(cm[np.arange(len(k)), k])[:, None]
        up = Fn.interpolate(cmk, size=(160, 160), mode='bilinear', align_corners=False)[:, 0].clamp(min=0)
        inside = up[:, 24:136, 24:136].sum((1, 2)) / up.sum((1, 2)).clamp(min=1e-6)
        share += inside.tolist(); pred += k.tolist()
    share = np.array(share); y = Y[test]
    out['inside_share_by_true_class'] = {CL[c]: {'median': float(np.median(share[y == c])), 'mean': float(share[y == c].mean())} for c in range(4)}
    out['inside_share_all'] = {'median': float(np.median(share)), 'mean': float(share.mean())}
    out['box_area_share'] = (112 * 112) / (160 * 160)
    out['int8_acc'] = float((np.array(pred) == y).mean())
    print(json.dumps(out, indent=1))
json.dump(out, open('../results/cam_share.json', 'w'), indent=1)
