# RoboticCC 2027 manuscript-overhaul prompt

Use the following prompt with an online ChatGPT agent that has access to the
GitHub repositories. The prompt deliberately distinguishes editable targets
from read-only evidence sources.

---

Act as a senior computer-vision and robotic-perception researcher preparing a
complete, publication-quality regular paper for **RoboticCC 2027: the 10th
International Conference on Robotic Computing & Communication (formerly the
IEEE International Conference on Robotic Computing/IRC)**. Work directly in
the connected GitHub repositories. Do not stop at recommendations: inspect the
evidence, edit the LaTeX and assets, compile the manuscript, visually audit the
PDF, commit the changes, push the branch, and report the resulting commit.

## Editable target

Edit only:

- repository: `EXPX3/confxxx_checkbranch`
- branch: `ieeercc2027`
- manuscript entry point: `main.tex`
- expected local checkout when available:
  `/datatank/giridhar.vb/repos/ws_jepa_occufly_test/confxxx_checkbranch_ieeercc2027`

The ACCV manuscript is **read-only evidence**. Do not edit, commit, reformat,
or push any file in `ws_jepa_occufly_test_accv26/confxxx_checkbranch` or its
`accv26-support` branch.

## Venue constraints: apply these before all writing preferences

Follow the official RoboticCC 2027 submission instructions at
<https://www.roboticcomputing.org/submission> and the supplied IEEE files in
the target repository.

- Use `\documentclass[10pt,conference,letterpaper]{IEEEtran}`.
- Prepare a regular paper in the official IEEE double-column format.
- The venue's standard submission limit is **8 pages total, including text,
  figures, tables, and references**.
- Do not use an appendix as an uncounted extension; all manuscript content
  counts toward the limit.
- The conference permits at most two paid extra pages. The requested output in
  this task is nevertheless an **expanded working draft with exactly 12 body
  pages, excluding references and any appendix**, plus no more than one page of
  references. Label it clearly as non-submission-compliant and preserve enough
  provenance for a later venue-length reduction.
- Preserve IEEE font sizes, margins, column widths, and spacing. Never compress
  the paper through negative spacing, tiny captions, scaled body text, or
  template modifications.
- Submit-ready output must be PDF. Keep author metadata exactly as found unless
  verified details and explicit instructions are available; never invent
  authors, affiliations, acknowledgements, funding, or conflicts.

Do not add an appendix merely to evade either limit. The user's 12-body-page
working-draft instruction controls this editing task; the eight-page venue rule
controls the later submission reduction.

## Read-only source repositories and evidence priority

Use the following as evidence, but modify only the target branch:

1. `EXPX3/ws_jepa_occufly_test`, branch `voxDet_NeurIPS_25`, including its
   README, reports, result CSV/JSON files, checkpoints, and experiment code.
2. `EXPX3/ws_jepa_occufly_test`, branch `accv26-support`, for additional
   DINOv3 depth-probe implementation and traceability material.
3. `EXPX3/confxxx_checkbranch`, branch `wacv27-support`, for the fuller WACV
   narrative, figures, method descriptions, and corrected result tables.
4. `EXPX3/confxxx_checkbranch`, branch `accv26-support`, only as historical
   manuscript evidence. It is strictly read-only.
5. `reports/voxdet_foundationssc_comparison_20261005/report.pdf` and the
   machine-readable files that generated it, especially its class-wise SSC
   comparison corresponding to Fig. 7.

Evidence hierarchy:

1. machine-readable evaluation artifacts and checkpoint metadata;
2. audited repository README/result ledgers;
3. current manuscript tables;
4. prose notes and run narratives.

When values conflict, resolve them from primary artifacts and document the
choice in the commit message or a private audit note. Never average across
incompatible protocols. Never convert a validation result into a test result.
Never fabricate, estimate, interpolate, or forward-fill an unfinished value.

## Scientific framing and terminology

Write as an established computer-vision/robotic-perception researcher. The
paper studies frozen V-JEPA 2.1 image/video representations and frozen DINOv3
image representations for aerial monocular depth estimation (MDE) and
depth-aware semantic scene completion (SSC) on OccuFly. The strongest completed
SSC treatment combines DINOv3 context, a frozen native MVSFormer++ multiview
depth provider, and VoxDet/VoxNT decoding.

Use publication-facing method labels, for example:

- `V-JEPA 2.1 + linear depth probe`
- `V-JEPA 2.1 + four-level DPT`
- `V-JEPA 2.1 + FiLM-DPT`
- `DINOv3 + linear depth probe`
- `DINOv3 + FiLM-DPT`
- `V-JEPA 2.1 + FiLM-DPT + VoxDet`
- `DINOv3 + FiLM-DPT + VoxDet`
- `DINOv3 + MVSFormer++ + VoxDet`

Do not expose implementation or run-management labels anywhere in the paper,
captions, figure legends, appendix, PDF metadata, or supplementary text. Search
for and remove terms such as `V07b`, `V5-V2`, `v5v2`, `v5v2dinov3`, numbered
`Variant` labels, run-directory names, checkpoint hashes, script names, DGX
hostnames, queue names, and internal split nicknames. Seeds may be identified
numerically only when reporting repeated-run statistics.

Expand each acronym at first use. Use MDE, SSC, SC IoU, mIoU, MVS, DPT, FiLM,
and VoxNT consistently after definition. Prefer precise claims such as
`is associated with`, `outperforms under the matched protocol`, or `persists
across three independent runs`; do not claim causality or statistical
significance without the necessary controlled repetitions.

## Required removal

Remove the altitude-interpolation ablation completely. Delete its table, all
main-text and appendix discussion, cross-references, contribution claims, and
conclusions. Search the entire source tree and rendered PDF for
`altitude interpolation`, `interpolation`, and labels associated with that
table. Do not replace it with another split-specific interpolation claim.
Altitude and camera-intrinsics conditioning in the standard scene-disjoint MDE
protocol may remain.

Do not discuss or name the coarse taxonomy-remapping experiment. Keep the
paper focused on the standard OccuFly taxonomy and analyze the long tail using
class supports and per-class IoU instead.

## MDE table and analysis

Extend the principal MDE table with the relevant, traceable methods while
keeping it legible in IEEE two-column format:

- V-JEPA 2.1 last-layer linear probe;
- V-JEPA 2.1 four-level DPT;
- V-JEPA 2.1 final-layer SFP/DPT, if complete and directly traceable;
- V-JEPA 2.1 FiLM-DPT;
- V-JEPA 2.1 temporal aggregation and pose-aware multiview diagnostics when
  completed and evaluated on a clearly stated compatible test subset;
- DINOv3 linear probe;
- DINOv3 FiLM-DPT;
- relevant published OccuFly baselines.

Report the evaluation population, units, direction arrows, and checkpoint
selection protocol. Keep pending methods out of numerical ranking; at most,
name genuinely pending results in a short non-numerical limitations/future-work
sentence. Resolve the older 4.621 m versus the newer traceable 4.770 m V-JEPA
FiLM-DPT result from primary artifacts and use one protocol-consistent value.
Do not preserve a known anomalous or unauditable failed DPT run as evidence.

## SSC table and repeated-run result

Replace the old SSC comparison with a compact protocol-aware table. Include
only variants that answer the paper's scientific questions:

- published OccuFly baselines;
- a representative monocular/context baseline;
- matched V-JEPA 2.1 monocular geometry with VoxDet;
- matched DINOv3 monocular geometry with VoxDet;
- the historical DINOv3--MVSFormer++ FoundationSSC control if its metric
  definition is stated and kept visually separate;
- the final DINOv3--MVSFormer++--VoxDet treatment over three independent runs.

Do not place incompatible semantic averages in one undifferentiated mIoU
column. If needed, use two panels or separate columns with explicit definitions.
Use the fixed-denominator score as the primary repeated-run semantic metric.

The currently verified three-run aggregate for the final multiview treatment
is mean plus sample standard deviation (`ddof=1`):

- SC IoU: `49.3273 +/- 1.1708` percent;
- GT-present semantic mIoU: `5.6028 +/- 0.4256` percent;
- fixed-21 semantic mIoU: `4.8024 +/- 0.3648` percent;
- union-present semantic mIoU: `5.2154 +/- 0.3477` percent;
- SC precision: `62.9624 +/- 0.5836` percent;
- SC recall: `69.5109 +/- 2.3409` percent;
- validation-selected epochs: `13, 12, 11`.

The matched monocular seed-42/43/44 replications are also complete. Use the
machine-readable report in
`reports/paired_monocular_foundation_ssc_20261007/data/three_seed_metrics.csv`
and preserve its unequal-budget caveat for historical V-JEPA seed 42:

- V-JEPA 2.1 SC IoU: `44.2408 +/- 1.3497` percent;
- V-JEPA 2.1 fixed-21 mIoU: `3.6911 +/- 0.2405` percent;
- DINOv3 SC IoU: `47.2954 +/- 0.5009` percent;
- DINOv3 fixed-21 mIoU: `4.2945 +/- 0.4083` percent;
- paired DINOv3-minus-V-JEPA SC difference: `+3.0546 +/- 1.6150` points;
- paired fixed-21 difference: `+0.6033 +/- 0.6454` points.

V-JEPA selected epochs are `2, 13, 12`; DINOv3 selected epochs are `9, 6, 5`.
The historical V-JEPA seed-42 history contains epochs 0--8, while the other
five histories contain epochs 0--19. Do not call this a fully equal-budget
three-pair comparison or a significance result.

Recompute these values from primary artifacts before publication. Include a
newer run only if it has a complete, auditable evaluation under the same
protocol. SC precision and recall are standard and useful here because they
show whether occupied-space gains arise through overprediction; define them
once and report them compactly alongside SC IoU.

Define all SSC averages explicitly:

- SC IoU: binary occupied-versus-empty intersection over union;
- GT-present mIoU: average over occupied classes present in ground truth;
- union-present mIoU: average over occupied classes present in ground truth or
  predictions, retained only for historical compatibility;
- fixed-21 mIoU: average over the fixed 21 occupied classes, with zero
  contribution for absent/failed classes as defined by the audited evaluator.

Never encode a class with zero test ground-truth support as a successful zero
IoU. Display it as `N/A` or `--` and state the denominator policy.

## Class-wise and long-tail evidence

Include a compact class-wise SSC comparison for selected, scientifically
important methods, using the report figure only as a visual reference and the
underlying CSV/JSON data as the numerical source. At minimum compare the final
multiview method against a relevant monocular method and one established
control when protocol-compatible.

Use one information-dense, IEEE-readable long-tail figure rather than a large
body table:

- sort occupied classes by training-set voxel support;
- top/aligned panel: train, validation, and test voxel counts on a logarithmic
  axis, with split totals or normalized frequency clearly identified;
- bottom/aligned panel: per-class IoU for the selected SSC methods, with the
  final method shown as mean plus variability across the three runs;
- preserve identical class order across panels;
- use `N/A` for classes absent from test ground truth;
- use a colorblind-safe palette, readable typography, and vector output when
  practical.

The discussion should distinguish class absence from model failure and relate
support to performance without claiming that frequency alone causes error.
Audit the current evidence that person, bicycle, and cable are present but have
zero IoU in all three final runs, while water, cable tower, and crane are absent
from test ground truth. State this only if the machine-readable confusion
matrices confirm it.

## Whole-paper overhaul

Produce a coherent paper, not a concatenation of the ACCV and WACV drafts.

1. **Title and abstract:** emphasize aerial robotic perception, frozen
   foundation representations, controlled MDE-to-SSC transfer, and the final
   three-run result. Avoid acronyms in the title. In the abstract, report only
   final verified test results and one clear conclusion.
2. **Introduction:** motivate metric geometry, free-space reasoning, and
   semantic occupancy for UAV navigation and planning. State a focused research
   question and contributions supported by completed evidence.
3. **Related work:** synthesize aerial MDE, self-supervised image/video
   foundations, calibrated multiview geometry, and camera-based SSC. Do not turn
   it into a method catalog.
4. **Method:** define frozen feature extraction, linear/DPT/SFP decoding,
   FiLM camera conditioning, temporal aggregation, MVSFormer++ geometry,
   probability-aware lifting, VoxDet/VoxNT decoding, and trainable versus frozen
   modules. Keep implementation minutiae out unless necessary for
   reproducibility.
5. **Experimental setup:** state OccuFly splits, resolutions, depth range,
   voxel grid, class protocol, training budget, validation-only checkpoint
   selection, repeated-run design, and metrics. Separate test results from
   validation diagnostics.
6. **Results:** organize around scientific questions: representation and
   decoder effects for MDE; camera metadata and temporal/multiview geometry;
   transfer to SSC; repeated-run stability; class-wise long-tail behavior.
7. **Discussion and limitations:** explain descriptive versus causal evidence,
   single-dataset scope, the lack of onboard latency/energy measurements,
   incomplete repetition for architectural controls, and the distinction
   between completion and semantic recognition.
8. **Conclusion:** answer the research question with completed evidence only.
   Do not introduce new numbers or pending claims.

Retain only figures and tables that materially support the argument. Prefer an
architecture overview, one compact MDE table, one compact SSC table, one
long-tail/class-wise figure, and at most one qualitative panel. Do not crowd the
eight-page IEEE paper with exhaustive seed tables; report aggregates in the
body and preserve exact per-run numbers in repository evidence unless space
remains within the same eight-page limit.

## Citation and V-JEPA 2.1 densification requirement

When explaining why the final V-JEPA 2.1 layer can support dense prediction,
cite the V-JEPA 2.1 paper specifically for dense prediction over masked and
visible tokens and deep self-supervision that propagates local information to
the output representation. The current audited citation points to Sections
2.3.1--2.3.2 and Appendix 9.1 of the V-JEPA 2.1 paper. Verify the paper and BibTeX
record before preserving those pinpoint references. Do not overstate the
mechanism beyond the source.

Use primary papers and official project/dataset sources for technical claims.
Use IEEE numeric citation style through `IEEEtran.bst`. Check every cited key,
remove uncited entries only when safe, and do not invent bibliographic fields.

## Validation and delivery

Before committing:

1. fetch the latest remote state and work from `ieeercc2027`;
2. ensure no ACCV file changed;
3. compile from a clean auxiliary state with the repository's documented
   LaTeX/BibTeX workflow or Tectonic;
4. verify that the final PDF is exactly eight pages total, including references;
5. visually inspect every page for clipped text, illegible plots, overlapping
   floats, bad column breaks, orphan headings, and malformed citations;
6. confirm `letterpaper`, IEEE two-column layout, and unmodified template
   geometry;
7. search source and extracted PDF text for forbidden internal labels and for
   all altitude-interpolation/remapping references;
8. recompute repeated-run mean and sample standard deviation from primary data;
9. verify metric denominators, units, arrows, split names, and absent-class
   handling;
10. ensure no pending value is rendered as a result and no author information
    is fabricated.

Commit the overhaul on a new branch based on `ieeercc2027`, preferably
`paper/roboticcc27-overhaul`, push it, and open a pull request into
`ieeercc2027` if the connection supports PR creation. In the final report give:

- branch, commit hash, and PR link;
- final PDF page count;
- tables/figures added, removed, or consolidated;
- exact source artifacts used for every new numerical result;
- unresolved evidence gaps;
- confirmation that the ACCV manuscript was not modified;
- confirmation that the altitude-interpolation and coarse-remapping analyses
  are absent from source and rendered PDF;
- confirmation that internal implementation names are absent from the paper.

---
