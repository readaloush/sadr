"""Build an index of the dataset with file metadata, patient ids and perceptual hashes."""
import glob, os, re, json, numpy as np
from PIL import Image
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
CL = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
def patient(cls, fn):
    m = re.match(r'person(\d+)_', fn)
    if m: return 'kp' + m.group(1)                 # Kermany pneumonia patients
    m = re.match(r'(NORMAL2-)?IM-(\d+)-', fn)
    if m: return 'kn' + m.group(2)                 # Kermany normal patients
    return None
def phash(im):
    a = np.asarray(im.convert('L').resize((32, 32), Image.BILINEAR), np.float32)
    from scipy.fft import dctn
    d = dctn(a, norm='ortho')[:8, :8].ravel(); return ''.join('1' if v > np.median(d[1:]) else '0' for v in d)
rows = []
for f in sorted(glob.glob(ROOT + '/*/*/*')):
    split, cls, fn = f.split('/')[-3:]
    im = Image.open(f)
    rows.append({'path': f'{split}/{cls}/{fn}', 'split': split, 'y': CL.index(cls), 'ext': os.path.splitext(fn)[1].lower(), 'mode': im.mode,
                 'w': im.size[0], 'h': im.size[1], 'bytes': os.path.getsize(f), 'patient': patient(cls, fn), 'phash': phash(im)})
json.dump(rows, open(os.path.join(ROOT, '..', 'index.json'), 'w'))
print(len(rows))
