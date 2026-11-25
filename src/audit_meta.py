"""How far can a classifier get without looking at the image? File type, colour mode, width, height and file size only."""
import json, numpy as np, lightgbm as lgb, collections
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
R = json.load(open('../data/index.json'))
EXT = ['.jpg', '.jpeg', '.png']; MODE = ['L', 'RGB', 'RGBA', 'P']
X = np.array([[EXT.index(r['ext']), MODE.index(r['mode']), r['w'], r['h'], r['w']/r['h'], np.log(r['bytes'])] for r in R], float)
y = np.array([r['y'] for r in R]); sp = np.array([r['split'] for r in R])
tr, te = sp != 'test', sp == 'test'
out = {}
for name, cols in {'size_only (w,h,aspect)': [2, 3, 4], 'file_type_only (ext,mode)': [0, 1], 'all_metadata': list(range(6))}.items():
    m = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(X[tr][:, cols], y[tr]); p = m.predict(X[te][:, cols])
    out[name] = {'acc': accuracy_score(y[te], p), 'macro_f1': f1_score(y[te], p, average='macro'), 'cm': confusion_matrix(y[te], p).tolist()}
    print(name, round(100*out[name]['acc'], 1), 'macroF1', round(out[name]['macro_f1'], 3), out[name]['cm'])
# patient overlap for the two Kermany classes
pat = collections.defaultdict(set)
for r in R:
    if r['patient']: pat[r['patient']].add(r['split'])
both = [p for p, s in pat.items() if 'test' in s and len(s) > 1]
print('Kermany patients', len(pat), 'in test and train/val', len(both))
out['kermany_patients'] = len(pat); out['kermany_patients_overlap'] = len(both)
json.dump(out, open('../results/audit_meta.json', 'w'), indent=1)
