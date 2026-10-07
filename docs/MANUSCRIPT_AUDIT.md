# RoboticCC working manuscript audit

## Scope and current user overrides

The editable repository is `EXPX3/confxxx_checkbranch`. The working branch is
`paper/roboticcc27-overhaul`, based on `ieeercc2027` at
`3eaebb63a614ab7990888ef8f75d5a2ea4f17fcc`.

The user explicitly requested **12 pages total**, with references occupying no
more than one page, to allow manual cutting to the venue limit. This overrides
the earlier eight-page working-deliverable instruction. The resulting version
is an expanded working draft, not an eight-page submission.

The subsequent instruction is to **report only standard 21-class semantic
mIoU**. The manuscript, captions, abstract, and results obey this. Other
original evaluator fields remain in raw evidence solely to preserve provenance;
they are not presented as alternative results in the paper. Historical semantic
scores without a verified standard rescore are omitted.

Only this manuscript checkout was edited. The experiment source repository,
the ACCV branch, and the WACV branch were read-only. The target base and the ACCV
remote heads were checked before publication. The official IEEE class and BST
are byte-identical to the target base.

## Figures and tables

- Figure 1: new vector architecture overview; frozen versus trainable modules,
  camera-conditioned depth, categorical MVS geometry, and occupancy output.
- Figure 2: new vector two-panel support/IoU figure, generated from saved class
  counts and three-run marginals. The ordering is descending **test** support.
  Training/validation histograms are unavailable; the caption and axis state
  that limitation. Water, cable tower, and crane are N/A; person, bicycle, and
  cable retain their observed zero-IoU failures.
- Table I: corrected image-depth records, a separate 3,830-target temporal/MVS
  diagnostic, and a published reference. No pending or anomalous run is ranked.
- Table II: compact SSC table containing only 21-class mIoU as the semantic
  average, plus binary SC IoU, precision, and recall where auditable.
- Exhaustive body seed/class tables are replaced by the aggregate and figure;
  exact saved records and generated class comparisons remain under `docs/evidence/`.

There is no added qualitative panel because no new, auditable prediction panel
was required to support the argument. Removed height-split and changed-taxonomy
analyses are absent from the paper sources and PDF. Internal run names appear
only in evidence/provenance files, never in publication text or figure metadata.

## Primary SSC provenance

The experiment implementation audit is anchored at
`EXPX3/ws_jepa_occufly_test:363b81093be6159bb0ec8de3f37911625d4e3880`,
branch `voxDet_NeurIPS_25`. The recovered report state is
`262a8557605e00d05d0c54bd27f3ac88d230c333`.

The three saved artifacts are copied without changing their original numeric
fields to `docs/evidence/multiview_run_0.json`, `multiview_run_1.json`, and
`multiview_run_2.json`. Original source paths, source SHA-256 values, checkpoint
paths, checkpoint SHA-256 values, and selected epochs are inside each record.
The source report paths are:

| Seed | Repository source |
| --- | --- |
| 0 | `reports/voxdet_foundationssc_comparison_20261005/data/07b_test_artifact_snapshot.json` |
| 1 | `reports/voxdet_foundationssc_comparison_20261005/data/07b_seed1_test_artifact_snapshot.json` |
| 2 | `reports/voxdet_foundationssc_comparison_20261005/data/07b_seed2_test_artifact_snapshot.json` |

`scripts/build_paper_assets.py` reconstructs integer semantic TP counts from
full-precision per-class IoU and GT/prediction marginals. The empty-class counts
also determine binary occupied TP/FP/FN. It checks each saved metric, identical
GT-support vectors, class absence, and the persistent zero-IoU claims, then
computes means and sample SD with `ddof=1`. The generated results are
`docs/evidence/recomputed_statistics.json` and `classwise_comparison.csv`.

The standard headline results are:

| Metric (%) | Mean | Sample SD |
| --- | ---: | ---: |
| SC IoU | 49.3273 | 1.1708 |
| 21-class mIoU | 4.8024 | 0.3648 |
| SC precision | 62.9624 | 0.5836 |
| SC recall | 69.5109 | 2.3409 |

The canonical local control/class figure source is
`experiments_wacv27/ssc/code/voxDet/corrected_evaluation_v1/report_bundle_20260828_v5v2_final/corrected_class_metrics.csv`,
recovered via `reports/voxdet_foundationssc_comparison_20261005/data/corrected_class_metrics.csv`
and preserved locally as `docs/evidence/canonical_class_counts.csv`. The recovered
and local CSV bytes match exactly. Control scores,
their precision/recall, test occupied total, empty fraction, building fraction,
and plotted per-class IoUs are recomputed from these counts. No training
frequency is inferred from them.

The fresh DINOv3 monocular result and historical multiview control are retained
in `docs/evidence/additional_evaluation_records.json`, copied from
`reports/voxdet_foundationssc_comparison_20261005/data/additional_reported_results.json`.
The monocular record includes all 3,842 targets and manifest/checkpoint hashes;
the multiview historical record lacks sufficient marginals for a standard
semantic rescore. Its semantic result is therefore omitted.

## Selection and gradient checks

The final multiview trainer actually selects strictly by its retained
validation `ssc_miou_legacy_union_present` field; it has no lexicographic
tie-breaker. The monocular contract selects validation GT-present mIoU. These
historical criteria are preserved here for exact traceability without listing
other semantic averages in the manuscript. Reporting standard 21-class test
mIoU does not retrospectively alter how a checkpoint was selected.

Selected multiview epochs are 13, 12, and 11. The V-JEPA monocular record was
selected at epoch two in a realized search through epoch eight, whereas the
DINOv3 monocular record selected epoch nine after a larger search budget. It
is labeled accordingly.

The final MVS provider and view-transformer parameters are frozen. The sparse
voxel-attention inputs are explicitly detached; context still trains through
the dense lift--splat/fusion route. This distinction follows the native
`vjepaF2Dfoundationssc_foundationssc_hybrid_original` adapter, not the older
monocular wrapper. The MVS posterior is projected onto 256 lifting centers over
0--64 m (0.125--63.875 m), rather than treating the image-depth 0.001--80 m range
as the lifting grid. Native MVS geometry has no SSC-to-depth gradient or depth
teacher loss.

## Retained MDE evidence and its limits

Every displayed MDE row is recorded in
`docs/evidence/mde_records.json`, with a repository ref and table/contract path.
Most values are retained from `EXPX3/confxxx_checkbranch:wacv27-support`,
`content/chapters/4_experiments.tex`, principal MDE table. They are **not** newly
recomputed primary pixel metrics. Primary `metrics.json`/`results-depth.csv`
files for these runs are missing from the connected branches.

The strict DINOv3 monocular frontend contract additionally records RMSE
4.358180 m, AbsRel 0.120620, and delta1 0.847647, plus the archived DPT checkpoint
SHA-256 `ebbf95d72db09c7067bbddc42d425fa84f305617569ab431cd21bc9365f24989`.
The paper retains the corrected V-JEPA FiLM-DPT value 4.770 m because it is
the current protocol-consistent corrected manuscript record. The older 4.621 m
value cannot be reconciled from primary files here and is not reused. The
anomalous old four-level DPT result is excluded; the corrected retained row is
6.730 m. SFP is described only as an implemented diagnostic, with no score.

The temporal attention + FiLM-MVS depth row is a retained completed record on
3,830 context-valid targets. Its 6.406 m RMSE is not a full-test monocular
comparison. Published DepthAnything2 depth and Symphonies/DISC occupancy values
are attributed to OccuFly. Published semantic values are omitted because no
standard-denominator rescore was established here.

## Bibliography verification

IEEE numeric citations use the unchanged `IEEEtran.bst`. Seventeen cited papers
remain; uncited scaffold entries are removed. Author-list abbreviation uses the
supported BST control without reducing reference font size.

- V-JEPA 2.1: `https://arxiv.org/html/2603.14482v1`, sections 2.3.1--2.3.2
  and **Appendix D.1**, correcting the stale Appendix 9.1 locator.
- DINOv3: `https://arxiv.org/abs/2508.10104`.
- OccuFly: `https://arxiv.org/html/2512.20770v1` and its official repository.
- MVSFormer++: `https://arxiv.org/abs/2401.11673`.
- FoundationSSC: `https://ojs.aaai.org/index.php/AAAI/article/view/37294`,
  AAAI 2026, 40(4), 3020--3028, DOI `10.1609/aaai.v40i4.37294`.
- VoxDet: final NeurIPS 2025 title/author/DOI record at
  `https://proceedings.neurips.cc/paper_files/paper/2025/hash/7478016a59b9851ff6685a3fdd0f6b2e-Abstract-Conference.html`.
- Symphonies and DISC: final CVF proceedings entries, CVPR 2024 pages
  20258--20267 and ICCV 2025 pages 26999--27009, respectively.

No unverified author, affiliation, acknowledgement, funding, or conflict is
inserted. The manuscript author block and PDF author metadata remain empty.

## Build and visual verification

`python3 scripts/compile_paper.py` builds in a new temporary auxiliary directory,
runs pdfLaTeX/BibTeX/pdfLaTeX/pdfLaTeX, checks letterpaper, exactly 12 pages,
references confined to page 12, unresolved citations/overfull boxes, forbidden
paper labels, and empty author metadata. It writes the ignored `main.pdf` and
`docs/evidence/layout_validation.json`. Every page is then rendered and visually
inspected, including both tables, vector figures, equations, and references.

Unresolved evidence gaps: primary MDE metric files; train/validation class
histograms; final multiview target manifests/sample counts; repeated historical
controls; matched completed monocular repetitions; inference/energy/latency
measurements; and uncertainty calibration. They are not filled with inferred
numbers, pending values, or claims of causal/statistical significance.
