# MVSFormer++ OccuFly results and reproducibility record

This file records the final test results used for the MVSFormer++ rebuttal
experiments. All paths are absolute so that the evaluations can be reproduced
from the shared `/datatank/giridhar.vb` workspace.

## 1. Standalone MVSFormer++ depth estimation (MDE)

The checkpoints were selected using validation performance and evaluated once
on the complete 3,842-reference OccuFly test split. Metrics are the mean of
per-reference metrics. There is **no median scaling and no ground-truth scale
alignment**. SILog is shown below in the fractional convention used in the
paper; the evaluator JSON stores it as a percentage (`12.968` and `19.254`).

| Backbone | δ1 ↑ | δ2 ↑ | δ3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE (m) ↓ | RMSE-log ↓ | MAE (m) ↓ | SILog ↓ | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 ViT-B/14 (original MVSFormer++ backbone) | 0.747903 | 0.838424 | 0.930279 | 0.231437 | 3.622671 | 7.656878 | 0.249196 | 5.605287 | 0.194722 | 0.221833 |
| DINOv3 ViT-H+/16 | 0.828477 | 0.937502 | 0.981950 | 0.136097 | 1.330885 | 4.610594 | 0.161511 | 3.314097 | 0.129682 | 0.250402 |
| V-JEPA 2.1 ViT-G/16 (image) | 0.761207 | 0.845204 | 0.930800 | 0.224937 | 3.517727 | 7.516732 | 0.244211 | 5.437209 | 0.192536 | 0.225785 |

All three evaluations use test split hash
`40b9d8db0fde742250171d7d64582446ae06419c9e6519c5cdaee0ef469a8943`.

### MDE checkpoints and outputs

| Variant | Validation-selected checkpoint | Checkpoint epoch | Full-test metrics |
|---|---|---:|---|
| DINOv2 | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/official_structure_occufly/seed0_screen_5epoch_ddp8/dinov2_base/seed_0/best.pt` | 0 | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/official_structure_occufly/seed0_screen_5epoch_ddp8/dinov2_base/seed_0/metrics_test_full.json` |
| DINOv3 | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/official_structure_occufly/seed0_screen_5epoch_ddp8/dinov3_hplus/seed_0/best.pt` | 1 | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/official_structure_occufly/seed0_screen_5epoch_ddp8/dinov3_hplus/seed_0/metrics_test_full.json` |
| V-JEPA 2.1 | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/official_structure_occufly/seed0_screen_5epoch_ddp8/vjepa21_vitG_image/seed_0/best.pt` | 0 | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/official_structure_occufly/seed0_screen_5epoch_ddp8/vjepa21_vitG_image/seed_0/metrics_test_full.json` |

The checkpoints came from the seed-0 five-epoch screening protocol. They are
not 15-epoch full-training checkpoints. This qualification must accompany the
reported results.

## 2. FSSC using the native MVSFormer++ posterior

Each FSSC variant uses a DPT context head and frozen MVSFormer++ depth from the
same backbone family. The final-stage categorical MVS posterior is projected
directly into FoundationSSC's 256 depth bins; it is not replaced by a Gaussian
around expected depth. Test evaluation uses the validation-best FSSC
checkpoint. SC IoU and SSC mIoU are reported as percentages.

| FSSC variant | Best FSSC epoch | SC IoU (%) ↑ | SSC mIoU (%) ↑ | Embedded depth RMSE (m) ↓ | Embedded depth AbsRel ↓ |
|---|---:|---:|---:|---:|---:|
| DINOv3 DPT context + DINOv3 MVS depth | 11 | **43.36698** | **4.47280** | 5.046839 | 0.125280 |
| V-JEPA 2.1 DPT context + V-JEPA 2.1 MVS depth | 4 | 35.02234 | 3.28268 | 8.355440 | 0.208831 |

### DINOv3 MVSFormer++ with VoxDet V5-V2

Replacing the control's FoundationSSC decoder/objective with the corrected
VoxDet V5-V2 decoder, long-tail loss, and full-resolution class-balanced VoxNT
selects epoch 13 by validation union-present SSC mIoU. The native MVS posterior
remains frozen and does not receive scheduled SSC-to-depth gradients. On the
canonical test split the treatment obtains **48.0635% SC
IoU**, **4.8426% union-present SSC mIoU**, and **5.1116% GT-present SSC mIoU**.
The union-present score is used for direct comparison with the 43.36698/4.47280
control. The complete test artifact is stored at
`data/checkpoints/my_checkpoints/07b_dinov3_mvsformerpp_voxdet_v5v2_seed0_20260929/test_best_epoch13/test_metrics.json`.

The embedded depth values above are those produced by the FSSC calibrated
dataset join/evaluator and therefore should not be substituted for the
standalone 3,842-reference MDE results in Section 1. The join excludes the
validated invalid-depth reference listed in the MVS manifest.

Additional test audit values:

| Variant | mIoU incl. empty (%) | Valid projection | Retained depth mass | Projected voxel fraction | Proposal fraction | Valid voxel fraction | Occupied voxel fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv3 | 8.46303 | 1.000000 | 0.999998 | 0.010528 | 0.012817 | 0.439038 | 0.065248 |
| V-JEPA 2.1 | 7.24520 | 1.000000 | 0.999629 | 0.011276 | 0.014023 | 0.439038 | 0.065248 |

### Per-class test IoU (%)

SSC mIoU excludes the empty class and averages the occupied classes present in
the test union. `cable_tower` has no test union and is therefore not averaged.

| Class | DINOv3 | V-JEPA 2.1 |
|---|---:|---:|
| empty | 88.2676 | 86.4958 |
| road | 5.2840 | 2.8856 |
| walkway | 2.9591 | 1.2161 |
| dirt | 1.4404 | 0.7462 |
| gravel | 2.0169 | 0.6429 |
| rock | 0.0340 | 0.0276 |
| grass | 6.7851 | 5.3721 |
| vegetation | 3.1739 | 2.7479 |
| tree | 5.7496 | 5.1565 |
| ground_obstacle | 1.4961 | 0.5575 |
| person | 0.0000 | 0.0000 |
| bicycle | 0.0048 | 0.0261 |
| vehicle | 9.2535 | 6.5424 |
| water | 0.0000 | 0.0000 |
| building | 41.6907 | 33.2839 |
| roof | 1.4817 | 1.7790 |
| cable | 0.0000 | 0.0000 |
| parking_lot | 6.3104 | 4.3730 |
| construction | 0.2692 | 0.2548 |
| crane | 0.0000 | 0.0000 |
| truck | 1.5065 | 0.0420 |

### FSSC checkpoints and raw results

| Variant | Validation-best FSSC checkpoint | Test summary | Per-class test IoU |
|---|---|---|---|
| DINOv3 | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/fssc_mvsformerpp_native/dinov3_dpt_context__dinov3_mvsformerpp_native_depth/seed_0/ssc_best.pt` | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/fssc_mvsformerpp_native/dinov3_dpt_context__dinov3_mvsformerpp_native_depth/seed_0/metrics_test.json` | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/fssc_mvsformerpp_native/dinov3_dpt_context__dinov3_mvsformerpp_native_depth/seed_0/per_class_iou_test.csv` |
| V-JEPA 2.1 | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/fssc_mvsformerpp_native/vjepa21_dpt_context__vjepa21_mvsformerpp_native_depth/seed_0/ssc_best.pt` | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/fssc_mvsformerpp_native/vjepa21_dpt_context__vjepa21_mvsformerpp_native_depth/seed_0/metrics_test.json` | `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/fssc_mvsformerpp_native/vjepa21_dpt_context__vjepa21_mvsformerpp_native_depth/seed_0/per_class_iou_test.csv` |

## 3. Implementation locations

### MVSFormer++ depth

- Experiment root: `/datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/posed_mvs_depth`
- Main model integration: `posed_mvs/models/foundation_mvsformerpp.py`
- Foundation-feature adapter: `posed_mvs/models/foundation_adapter.py`
- DINOv3 adapter: `posed_mvs/backbones/dinov3_hplus.py`
- V-JEPA 2.1 adapter: `posed_mvs/backbones/vjepa21_vitG_image.py`
- Camera geometry: `posed_mvs/geometry/projection.py`
- Depth metrics: `posed_mvs/metrics/depth.py`
- Trainer: `scripts/train.py`
- Distributed evaluator: `scripts/evaluate.py`
- DINOv2 configuration: `configs/dinov2_base_mvsformerpp.yaml`
- DINOv3 configuration: `configs/dinov3_hplus_mvsformerpp.yaml`
- V-JEPA configuration: `configs/vjepa21_vitG_image_mvsformerpp.yaml`
- Invalid-depth manifest: `configs/occufly_invalid_depth_references.json`
- Local official MVSFormer++ source: `third_party/MVSFormerPlusPlus`

### FSSC native-posterior integration

Package root:
`/datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/ssc/code/vjepaF2Dfoundationssc_foundationssc_hybrid_original`

- Frozen MVS provider and posterior projection:
  `vjepafoundationssc/mvsformerpp_native_depth.py`
- FSSC trainer integration:
  `vjepafoundationssc/train_vjepa_occufly_ssc.py`
- Strict preprojected 256-bin lift-splat path:
  `vjepafoundationssc/foundation_ssc/real_adapters.py`
- Posterior mass/moment and checkpoint-contract tests:
  `tests/test_mvsformerpp_native_depth.py`
- DINOv3 launch script: `run_fssc_mvsformerpp_native_dinov3.sh`
- V-JEPA launch/queue script: `run_fssc_mvsformerpp_native_queue.sh`
- Design and integrity record:
  `/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/fssc_mvsformerpp_native/report-source.md`

The FSSC bridge uses five calibrated views at 768×1024 for MVS, deposits each
adaptive final-stage hypothesis barycentrically into the neighboring centers
of the fixed 256-bin `[0,64,0.25]` FoundationSSC grid, excludes unsupported
mass, renormalizes supported mass, and uses posterior expectation only for the
dense-depth interfaces. The MVS branch is frozen and detached from SSC
optimization.

## 4. Re-running the evaluations

Environment paths:

- MVS evaluation Python:
  `/datatank/giridhar.vb/repos/.codex_envs/occufly_nyu_py311/bin/python`
- FSSC Python:
  `/datatank/giridhar.vb/repos/.codex_envs/occufly_foundationssc_py311_torch21/bin/python`
- Dataset:
  `/datatank/giridhar.vb/repos/datasets/OccuFly_Dataset`

### Re-run all three standalone MDE test evaluations on eight GPUs

```bash
cd /datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/posed_mvs_depth
bash scripts/run_mvsformerpp_test_metrics_dgx4.sh
```

The script evaluates DINOv2, DINOv3, and V-JEPA 2.1 sequentially with eight
DDP ranks, four workers per rank, the exact checkpoints listed above, and
writes `metrics_test_full.json` beside each checkpoint.

### Validate the MVS/FSSC bridge

```bash
cd /datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/ssc/code/vjepaF2Dfoundationssc_foundationssc_hybrid_original
/datatank/giridhar.vb/repos/.codex_envs/occufly_foundationssc_py311_torch21/bin/python \
  -m pytest -q tests/test_mvsformerpp_native_depth.py
```

The production launch scripts contain the complete argument sets for both
variants. They currently use `--force` and the original output directories, so
**do not run them unchanged when preserving the recorded artifacts**. Copy a
script and change its `OUT`/`VJ_OUT` to a new directory before retraining:

```bash
cd /datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/ssc/code/vjepaF2Dfoundationssc_foundationssc_hybrid_original
bash run_fssc_mvsformerpp_native_dinov3.sh
bash run_fssc_mvsformerpp_native_queue.sh
```

For test-only reevaluation, retain the launch script's complete model/data
arguments, add `--eval-only --checkpoint /absolute/path/to/ssc_best.pt`, and
write to a separate output directory. The trainer validates the backbone tag,
selected layers, foundation checkpoint, depth source, MVS variant, and MVS
checkpoint contract while loading.

## 5. Primary references

- MVSFormer++ paper (ICLR 2024): <https://openreview.net/pdf?id=wXWfvSpYHh>
- Official implementation: <https://github.com/maybeLx/MVSFormerPlusPlus>
- Official custom-data instructions:
  <https://github.com/maybeLx/MVSFormerPlusPlus/blob/main/README.md#test-on-your-own-data>
- Local implementation report:
  `/datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/posed_mvs_depth/IMPLEMENTATION_REPORT.md`
- OccuFly camera/data audit:
  `/datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/posed_mvs_depth/OCCUFLY_DATA_AUDIT.md`
- Environment record:
  `/datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/posed_mvs_depth/ENVIRONMENT.md`
