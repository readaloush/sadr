import json, numpy as np
from PIL import Image
R = json.load(open('../data/index.json')); S = 160
X = np.zeros((len(R), S, S), np.uint8)
for i, r in enumerate(R):
    X[i] = np.asarray(Image.open('../data/raw/' + r['path']).convert('L').resize((S, S), Image.BILINEAR))
np.savez('../data/img160.npz', X=X, y=np.array([r['y'] for r in R]), split=np.array([r['split'] for r in R]))
print(X.shape)
