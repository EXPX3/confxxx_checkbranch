#!/usr/bin/env python3
"""Recompute the published statistics and draw manuscript assets from saved evidence."""
from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs' / 'evidence'
FIG = ROOT / 'content' / 'images'

def check_close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'Metric mismatch: {actual} != {expected}')

def audit_runs():
    runs = []
    names = None
    with (DATA / 'canonical_class_counts.csv').open() as stream:
        classes = list(csv.DictReader(stream))
    baseline = [r for r in classes if r['model'].startswith('FSSC RGB context DPT')]
    mono = [r for r in classes if r['model'].startswith('FSSC-VoxDet V5 v2')]
    if len(baseline) != 22 or len(mono) != 22:
        raise ValueError('Missing 22-class comparison evidence')
    names = [r['class_name'].replace('_', ' ') for r in baseline]
    expected_gt = np.array([int(r['gt_support']) for r in baseline])
    for seed in range(3):
        source = json.loads((DATA / f'multiview_run_{seed}.json').read_text())
        g = np.array(source['per_class_gt_support'], dtype=np.int64)
        p = np.array(source['per_class_prediction_support'], dtype=np.int64)
        iou = np.array([0.0 if v is None else v for v in source['per_class_iou']])
        if not np.array_equal(g, expected_gt):
            raise ValueError('Test GT-support vectors differ')
        raw_tp = iou * (g.astype(float) + p.astype(float)) / (1.0 + iou)
        t = np.rint(raw_tp).astype(np.int64)
        if np.max(abs(raw_tp - t)) > 1e-5 or np.any(t > np.minimum(g, p)):
            raise ValueError('Non-integer or inconsistent reconstructed marginals')
        union = g + p - t
        exact_iou = t / np.maximum(union, 1)
        for a, b in zip(exact_iou, iou): check_close(float(a), float(b))
        binary_tp = int(g.sum() - g[0] - p[0] + t[0])
        binary_fp = int(g[0] - t[0])
        binary_fn = int(p[0] - t[0])
        metrics = {
            'sc_iou': binary_tp / (binary_tp + binary_fp + binary_fn),
            'gt_present_miou': float(exact_iou[1:][g[1:] > 0].mean()),
            'fixed_21_miou': float(exact_iou[1:].mean()),
            'union_present_miou': float(exact_iou[1:][union[1:] > 0].mean()),
            'sc_precision': binary_tp / (binary_tp + binary_fp),
            'sc_recall': binary_tp / (binary_tp + binary_fn),
        }
        keys = ['sc_iou', 'ssc_miou_gt_present', 'ssc_miou_all_21_zero_filled',
                'ssc_miou_legacy_union_present', 'sc_precision', 'sc_recall']
        for (name, value), key in zip(metrics.items(), keys): check_close(value, source[key])
        runs.append({'seed': seed, 'selected_epoch': source['checkpoint_epoch'],
                     'metrics_percent': {k: 100*v for k, v in metrics.items()},
                     'gt_support': g.tolist(), 'prediction_support': p.tolist(),
                     'semantic_tp': t.tolist(), 'per_class_iou_percent': (100*exact_iou).tolist(),
                     'gt_present_classes': int((g[1:] > 0).sum()),
                     'union_present_classes': int((union[1:] > 0).sum()),
                     'valid_evaluated_voxels': int(g.sum())})
    summary = {}
    for key in runs[0]['metrics_percent']:
        values = np.array([r['metrics_percent'][key] for r in runs])
        summary[key] = {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1))}
    per_class = np.array([r['per_class_iou_percent'] for r in runs])
    present = expected_gt > 0
    absent = [names[i] for i in range(1,22) if not present[i]]
    failed = [names[i] for i in range(1,22) if present[i] and np.all(per_class[:, i] == 0)]
    if absent != ['water', 'cable tower', 'crane'] or failed != ['person', 'bicycle', 'cable']:
        raise ValueError('Class-absence/failure claim changed')
    output = {'sample_sd_ddof': 1, 'n_runs': 3, 'runs': runs, 'summary_percent': summary,
              'class_names': names, 'test_absent_classes': absent,
              'zero_iou_present_classes_all_runs': failed,
              'per_class_mean_percent': per_class.mean(axis=0).tolist(),
              'per_class_sample_sd_percent': per_class.std(axis=0,ddof=1).tolist(),
              'exact_test_uid_equivalence_verified': False,
              'train_validation_histograms_available': False,
              'class_order_basis': 'descending audited test GT support'}
    (DATA / 'recomputed_statistics.json').write_text(json.dumps(output, indent=2)+'\n')
    with (DATA / 'classwise_comparison.csv').open('w') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['class','test_gt_voxels','foundation_ssc_iou_percent',
                         'monocular_voxdet_iou_percent','multiview_voxdet_mean_percent',
                         'multiview_voxdet_sample_sd_percent','test_gt_present'])
        for i in range(1,22):
            is_present = bool(present[i])
            writer.writerow([names[i],int(expected_gt[i]),
                             baseline[i]['iou_percent'] if is_present else 'N/A',
                             mono[i]['iou_percent'] if is_present else 'N/A',
                             output['per_class_mean_percent'][i] if is_present else 'N/A',
                             output['per_class_sample_sd_percent'][i] if is_present else 'N/A',is_present])
    return output, baseline, mono

def draw_long_tail(output, baseline, mono):
    g = np.array(output['runs'][0]['gt_support'])[1:]
    order = np.argsort(-g, kind='stable')
    names = np.array(output['class_names'][1:])[order]
    x = np.arange(21)
    fig, (top, bottom) = plt.subplots(2,1,figsize=(3.5,3.9),sharex=True,
                                     gridspec_kw={'height_ratios':[1,1.55]})
    top.bar(x,g[order],color='#777777',width=.64)
    top.set_yscale('log'); top.set_ylim(300,2e9)
    top.set_ylabel('GT voxels\n(log scale)',fontsize=7)
    top.set_title('(a) Class distribution based on voxel count\n'
                  'in the OccuFly dataset',fontsize=7.5,loc='left',pad=3)
    for j,index in enumerate(order):
        if g[index] == 0:
            top.text(j,500,'0',ha='center',va='bottom',fontsize=8)
    top.text(.99,.93,f'Occupied total: {int(g.sum()):,}',transform=top.transAxes,
             ha='right',va='top',fontsize=6.5)
    present = g[order] > 0
    with (DATA / 'table3_classwise_iou.csv').open() as stream:
        variants = list(csv.DictReader(stream))
    styles = [
        ('#CC79A7', 'D', '--'),
        ('#0072B2', 's', '-'),
        ('#009E73', '^', '-'),
        ('#E69F00', 'v', '-.'),
        ('#D55E00', 'o', '-'),
        ('#6A3D9A', 'P', ':'),
    ]
    labels = ['SSC-1', 'SSC-2', 'SSC-3', 'SSC-4', 'SSC-5']
    class_order = output['class_names'][1:]
    series = []
    for label, row in zip(labels, variants):
        values = np.array([np.nan if row[name] == 'N/A' else float(row[name])
                           for name in class_order])[order]
        values[~present] = np.nan
        series.append((label, values))
    oracle = json.loads((DATA / 'o5_gtgt_seed0_classwise.json').read_text())
    oracle_support = np.array(oracle['per_class_gt_support'], dtype=np.int64)
    if not np.array_equal(oracle_support, np.array(output['runs'][0]['gt_support'])):
        raise ValueError('O5 and plotted SSC methods have different GT support')
    oracle_iou = np.array([np.nan if value is None else 100 * float(value)
                           for value in oracle['per_class_iou']])
    if len(oracle_iou) != 22:
        raise ValueError('O5 class-IoU vector must contain empty plus 21 semantic classes')
    oracle_fixed21 = float(np.nan_to_num(oracle_iou[1:]).mean())
    check_close(oracle_fixed21, 100 * oracle['ssc_miou_all_21_zero_filled'])
    oracle_values = oracle_iou[1:][order]
    oracle_values[~present] = np.nan
    series.append(('O5', oracle_values))
    for (label, values), (color, marker, linestyle) in zip(series, styles):
        bottom.plot(x, values, marker=marker, linestyle=linestyle, color=color,
                    markersize=2.4, linewidth=.75, label=label)
    for j,index in enumerate(order):
        if g[index] == 0:
            bottom.axvspan(j-.4,j+.4,color='#EEEEEE',zorder=-1)
            bottom.text(j,3,'N/A',ha='center',va='bottom',fontsize=8,rotation=90)
    bottom.set_ylabel('Per-class IoU (%)',fontsize=7)
    bottom.set_ylim(-1.5,57);bottom.set_xlim(-.6,20.6)
    bottom.set_title('(b) Semantic completion',fontsize=7.5,loc='left')
    bottom.set_xticks(x, names, rotation=65,ha='right',fontsize=5.7)
    bottom.legend(loc='upper right',fontsize=4.4,frameon=False,handlelength=1.25,
                  labelspacing=.25,borderaxespad=.25)
    for ax in (top,bottom):
        ax.tick_params(axis='y',labelsize=6)
        ax.grid(axis='y',alpha=.25,linewidth=.5)
        ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(left=.15,right=.985,top=.89,bottom=.245,hspace=.3)
    fig.savefig(FIG/'long_tail.pdf',bbox_inches='tight',pad_inches=.02,
                metadata={'Title':'OccuFly class distribution and class-wise semantic occupancy','Author':''})
    fig.savefig(FIG/'long_tail.png',dpi=220,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)

def draw_table3_heatmap():
    with (DATA / 'table3_classwise_iou.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5:
        raise ValueError('Table 3 heatmap must contain exactly five non-reference methods')
    classes = list(rows[0].keys())[1:]
    values = np.array([[np.nan if row[name] == 'N/A' else float(row[name])
                        for name in classes] for row in rows])
    expected_fixed21 = np.array([1.1774, 2.9870, 3.691132511833303,
                                 4.294452964306612, 4.802439257300768])
    observed_fixed21 = np.nan_to_num(values).mean(axis=1)
    if not np.allclose(observed_fixed21, expected_fixed21, rtol=0, atol=5e-5):
        raise ValueError('Per-class rows do not reproduce the Table 3 fixed-21 means')
    labels = ['SSC-1', 'SSC-2', 'SSC-3', 'SSC-4', 'SSC-5']
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    cmap = plt.colormaps['YlGnBu'].copy()
    cmap.set_bad('#dedede')
    image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=cmap,
                      norm=PowerNorm(.35, vmin=0, vmax=45))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                text, color = 'NA', '#192d45'
            else:
                text = f'{value:.2f}'
                color = 'white' if value > 8 else '#192d45'
            ax.text(j, i, text, ha='center', va='center', fontsize=3.25, color=color)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha='right', fontsize=4.7)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.set_ylabel('Table 3 SSC ID', fontsize=6.2)
    ax.tick_params(length=1.5, pad=1)
    ax.set_title('OccuFly test per-class IoU (%)', fontsize=7.5, weight='bold', pad=4)
    colorbar = fig.colorbar(image, ax=ax, fraction=.032, pad=.015)
    colorbar.set_label('IoU (%), nonlinear', fontsize=5.5)
    colorbar.ax.tick_params(labelsize=4.8)
    fig.subplots_adjust(left=.105, right=.94, top=.89, bottom=.35)
    fig.savefig(FIG / 'table3_per_class_iou_matrix.pdf',
                bbox_inches='tight', pad_inches=0,
                metadata={'Title': 'Table 3 per-class IoU matrix', 'Author': ''})
    fig.savefig(FIG / 'table3_per_class_iou_matrix.png', dpi=240,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_architecture():
    fig,ax=plt.subplots(figsize=(7.16,3.3))
    ax.set_xlim(0,7.16);ax.set_ylim(0,3.3);ax.axis('off')
    frozen='#D9EAF4'; trained='#F9DFC9'; neutral='#EEEEEE';edge='#444444'
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.04',
                     facecolor=color,edgecolor=edge,linewidth=.7))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8.4)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,linewidth=.8,color=edge))
    ax.text(.04,3.1,'(a) Frozen representations to metric depth',fontsize=10,weight='bold')
    box(.08,2.28,.8,.55,'Target RGB\n384 × 384',neutral)
    box(1.16,2.28,1.42,.55,'V-JEPA 2.1 / DINOv3\nfrozen encoder',frozen)
    box(2.89,2.28,1.75,.55,'Linear / DPT / SFP\ntrainable depth head',trained)
    box(5.15,2.28,1.7,.55,'256-bin distribution\nmetric depth expectation',neutral)
    for a,b in [((.92,2.55),(1.11,2.55)),((2.62,2.55),(2.84,2.55)),((4.68,2.55),(5.1,2.55))]:arrow(a,b)
    ax.text(3.76,2.95,'Altitude + intrinsics → FiLM',ha='center',fontsize=8.4)
    arrow((3.76,2.90),(3.76,2.85))
    ax.text(.04,1.96,'(b) Final multiview semantic scene completion treatment',fontsize=10,weight='bold')
    box(.08,1.02,.8,.55,'Target RGB\n384 × 384',neutral)
    box(1.16,1.02,1.16,.55,'DINOv3\nfrozen encoder',frozen)
    box(2.61,1.02,1.05,.55,'DPT context\ntrainable',trained)
    box(3.95,1.02,1.16,.55,'Probability-aware\nlifting / fusion',frozen)
    box(5.40,1.02,1.43,.55,'VoxDet + VoxNT\ntrainable decoder',trained)
    for a,b in [((.92,1.3),(1.11,1.3)),((2.36,1.3),(2.56,1.3)),((3.70,1.3),(3.90,1.3)),((5.15,1.3),(5.35,1.3))]:arrow(a,b)
    box(.08,.12,1.78,.55,'Five calibrated RGB views\n768 × 1024 + poses',neutral)
    box(2.14,.12,2.7,.55,'Native DINOv3 MVSFormer++\nfrozen categorical posterior',frozen)
    arrow((1.9,.4),(2.09,.4));arrow((4.53,.71),(4.53,.98))
    ax.text(5.98,.51,'Empty + 21 occupied classes\n192 × 128 × 128 voxels',ha='center',fontsize=8.4)
    arrow((6.11,.98),(6.11,.75))
    ax.text(5.82,1.82,'Training: occupancy / semantics /\nfull-resolution directional offsets',ha='center',fontsize=8.1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    fig.savefig(FIG/'architecture.pdf',metadata={'Title':'Frozen-feature depth and occupancy study','Author':''})
    fig.savefig(FIG/'architecture.png',dpi=220)
    plt.close(fig)

if __name__ == '__main__':
    FIG.mkdir(parents=True,exist_ok=True)
    output, baseline, mono = audit_runs()
    draw_long_tail(output,baseline,mono)
    draw_table3_heatmap()
    draw_architecture()
    print(json.dumps(output['summary_percent'],indent=2))
