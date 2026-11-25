# Sadr صدر

Four-class chest X-ray classification (COVID-19, normal, pneumonia, tuberculosis): how much of the accuracy comes from the lungs, and how small the model can get.

**Demo:** [readaloush.github.io/sadr](https://readaloush.github.io/sadr/): three 8-bit models run in your browser. Press *Hide the lungs* to run a model that was trained without ever seeing them.

Research and education only. Not a medical device.

## 1. What the accuracy is made of

The data is the Kaggle set "Chest X-ray (Pneumonia, COVID-19, Tuberculosis)": 7,135 images, 6,326 train, 38 validation, 771 test. Each class comes from a different collection, and it shows in the files themselves:

| Class | Files | Size |
|---|---|---|
| COVID-19 | 576 JPG, many sources | 400 to 3,000 px, mixed colour modes |
| Normal | 1,583 JPEG | about 1,500 × 1,200 |
| Pneumonia | 4,273 JPEG | about 1,000 × 650 |
| Tuberculosis | 703 PNG | all 512 × 512 |

Normal and pneumonia images come from the pediatric set of Kermany et al. (2018), children aged one to five from one hospital in Guangzhou. On the 771 test images:

| Classifier | Accuracy | COVID-19 found | TB found |
|---|---|---|---|
| LightGBM on file type and colour mode only, no pixels | 69.6% | 106 / 106 | 41 / 41 |
| LightGBM on image width, height and aspect ratio only | 77.6% | 74 / 106 | 41 / 41 |
| ResNet-18 trained and tested with the centre 70% of every image blacked out | 84.6% | 91 / 106 | 41 / 41 |
| ResNet-18 on the whole image | **92.0%** | 103 / 106 | 41 / 41 |

A model that never sees the lungs still finds every tuberculosis case and 86% of COVID-19 cases. The labels are confounded with the source, so on this data a high score mostly measures recognising the collection an image came from (`src/audit_meta.py`, `src/teacher.py masked`). Images whose lungs are hidden at test time but which a normally trained model then classifies fall to 27.6%; the comparison above uses a model trained the same way it is tested.

Normal against pneumonia is the only pair from the same source, so it is the part that tests the lungs: 90.5% on the test images, 98.7% on a validation split cut at random from the training images. The often-reported ~98% on this set is a validation-split number.

Exact copies between test and train: none. Near-duplicates (NCC ≥ 0.95 on 64 × 64 thumbnails): 11 test images, all with the same label (`src/audit_dupes.py`). File names in the pneumonia class reuse "person" numbers in train and test, but such pairs are no more similar than random pairs, so they are not taken as the same patients.

## 2. Making it small

Teacher: ResNet-18 from ImageNet weights, 160 × 160 grayscale (repeated to three channels), 6 epochs, AdamW with a one-cycle schedule, light affine and brightness augmentation, class-weighted cross-entropy, 2 CPU threads. Validation: a stratified 10% of train + val. The 771 test images are scored once per model.

Compression:

- **Pruning**: structured L1-magnitude channel pruning with [torch-pruning](https://github.com/VainF/Torch-Pruning) at 30%, 50% and 70% of channels (whole filters removed, residual connections kept consistent), then 3 epochs of fine-tuning (`src/prune.py`).
- **Distillation**: the same fine-tuning with a KL loss to the teacher (T = 4, α = 0.7).
- **Quantization**: ONNX Runtime static int8, QDQ, per-channel weights, 256 calibration images (`src/quant_bench.py`).

| Model | Params | MACs | ONNX size | Latency | Test accuracy (95% CI) | vs teacher |
|---|---|---|---|---|---|---|
| ResNet-18, fp32 | 11.18 M | 929 M | 42.6 MB | 14.2 ms | 92.0 (90.0–93.8) | |
| ResNet-18, int8 | | | 10.8 MB | 4.4 ms | 92.0 (90.0–93.8) | 0.0 |
| 30% pruned, fp32 | 5.46 M | 462 M | 20.8 MB | 8.1 ms | 89.9 (87.5–92.0) | −2.1 (−3.4, −0.9) |
| 50% pruned, fp32 | 2.80 M | 248 M | 10.7 MB | 3.4 ms | 90.0 (87.9–92.1) | −1.9 (−3.4, −0.6) |
| 70% pruned, fp32 | 1.00 M | 96 M | 3.8 MB | 2.6 ms | 89.8 (87.7–91.8) | −2.2 (−3.8, −0.8) |
| 70% pruned, int8 | | | **1.0 MB** | **1.2 ms** | 89.0 (86.9–91.2) | −3.0 (−4.5, −1.6) |

Latency: batch 1, one thread, ONNX Runtime CPU on an x86 server, median of 200 runs. Confidence intervals: bootstrap over test images; "vs teacher" is a paired bootstrap of the difference, in points (`src/stats.py`). Pruned rows without distillation; full results, including the distilled and int8 versions of every pruned model, are in `results/quant_bench.json` and `results/stats.json`.

What this shows:

- **int8 is free here**: four times smaller, three times faster, the same predictions within the test set's accuracy.
- **Pruning costs 2–3 points, and the cost barely depends on how much is pruned**: 30% and 70% land within a point of each other. Pruning 70% of channels and quantizing gives a model 42 times smaller and 12 times faster than the fp32 teacher.
- **Distillation did not help**: 89.1–90.8% with it against 89.8–90.0% without, within noise. One seed per setting.
- Pruning 30% of channels halves the parameters, but latency fell only from 14.2 to 8.1 ms; the gain grows with the ratio.

In a browser (ONNX Runtime Web, WebAssembly, one thread) on a headless x86 machine, the int8 teacher took 41.7 ms and the 70%-pruned int8 model 6.4 ms per image. The demo measures the same on the visitor's device.

## Demo

`docs/index.html` loads the int8 teacher (10.8 MB), the 70%-pruned int8 model (1.0 MB) and the int8 lungs-hidden model (10.8 MB); the two larger files are split into parts under 3.5 MB and joined in the browser. Images are resized to 160 × 160 in the browser; on the eight sample images the browser's labels match Python's. The samples are test images from the pediatric pneumonia set only (Kermany et al., CC BY 4.0).

## Run it

```
pip install -r requirements.txt
# put the Kaggle dataset's train/ val/ test/ folders under data/raw/
cd src
python meta.py && python cache.py
python audit_meta.py && python audit_dupes.py
python teacher.py full && python teacher.py masked
for r in 0.3 0.5 0.7; do python prune.py $r ce; python prune.py $r kd; done
python quant_bench.py && python stats.py && python export_demo.py
```

ImageNet weights: torchvision `resnet18-f37072fd.pth`.

## Limitations

- One seed per model; differences of a point or two between pruned variants are within noise.
- 160 × 160 input, chosen to train on CPU.
- The source confound cannot be removed from this dataset; a disease classifier needs every class from the same hospitals.

## References

- D. Kermany et al., "Identifying medical diagnoses and treatable diseases by image-based deep learning," *Cell*, 172(5), 2018. Data: Mendeley Data, CC BY 4.0.
- A. J. DeGrave, J. D. Janizek and S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nature Machine Intelligence*, 3, 2021.
- G. Fang et al., "DepGraph: Towards any structural pruning," *CVPR*, 2023 (torch-pruning).
- G. Hinton, O. Vinyals and J. Dean, "Distilling the knowledge in a neural network," 2015.

MIT License. Read Alalloş, [readaloush.com](https://readaloush.com).
