import sys
from common import *
mode = sys.argv[1]   # 'full' or 'masked'
m = train_model(resnet18(), epochs=6, mask=(mode == 'masked'))
torch.save(m.state_dict(), f'../results/teacher_{mode}.pt')
r = {'val': metrics(val, predict(m, val, mask=(mode == 'masked'))), 'test': metrics(test, predict(m, test, mask=(mode == 'masked')))}
json.dump(r, open(f'../results/teacher_{mode}.json', 'w'), indent=1)
print(mode, 'TEST acc', round(r['test']['acc'], 4), 'macroF1', round(r['test']['macro_f1'], 3), 'N-vs-P', round(r['test']['normal_vs_pneumonia_acc'], 4), r['test']['cm'], flush=True)
