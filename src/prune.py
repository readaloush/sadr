"""Structured channel pruning (L1 magnitude, torch-pruning) of the full teacher, then fine-tuning with or without distillation."""
import sys, copy, torch_pruning as tp
from common import *
ratio = float(sys.argv[1]); kd = sys.argv[2] == 'kd'; ep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
teacher = resnet18(False); teacher.load_state_dict(torch.load('../results/teacher_full.pt')); teacher.eval()
m = copy.deepcopy(teacher); ex = torch.randn(1, 3, 160, 160)
base_macs, base_params = tp.utils.count_ops_and_params(m, ex)
pr = tp.pruner.MagnitudePruner(m, ex, importance=tp.importance.MagnitudeImportance(p=1), pruning_ratio=ratio, ignored_layers=[m.fc])
pr.step()
macs, params = tp.utils.count_ops_and_params(m, ex)
before = metrics(val, predict(m, val))
print(f'ratio {ratio}: params {params/1e6:.2f}M ({params/base_params:.1%}), MACs {macs/1e6:.0f}M ({macs/base_macs:.1%}); val acc right after pruning {before["acc"]:.4f}', flush=True)
m = train_model(m, epochs=ep, lr=5e-4, teacher=teacher if kd else None)
tag = f'pruned{int(ratio*100)}_{"kd" if kd else "ce"}'
torch.save(m, f'../results/{tag}.pth')
r = {'ratio': ratio, 'kd': kd, 'params': params, 'macs': macs, 'base_params': base_params, 'base_macs': base_macs, 'val_after_prune': before,
     'val': metrics(val, predict(m, val)), 'test': metrics(test, predict(m, test))}
json.dump(r, open(f'../results/{tag}.json', 'w'), indent=1)
print(tag, 'TEST acc', round(r['test']['acc'], 4), 'macroF1', round(r['test']['macro_f1'], 3), 'N-vs-P', round(r['test']['normal_vs_pneumonia_acc'], 4), flush=True)
