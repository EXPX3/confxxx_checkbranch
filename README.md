# RoboticCC 2027 Paper Scaffold

This checkout targets the 10th International Conference on Robotic Computing
& Communication (RoboticCC 2027, formerly the IEEE International Conference
on Robotic Computing/IRC). It was created from `main` of
`EXPX3/confxxx_checkbranch` on the `ieeercc2027` branch. The existing
`content/` layout is deliberately retained.

The root `main.tex` now uses the official IEEE conference class and bibliography
style. The unmodified official source files from the supplied IEEE archives are
included locally so the manuscript can be compiled reproducibly without relying
on a system-installed version.

## Current expanded working draft

The user requested a **12-page total working version**, with references on no
more than one page, for manual reduction to the venue limit. This overrides the
earlier eight-page deliverable target for this revision. The only semantic
average reported in the paper is **standard 21-class mIoU**.

Build and verify it with `python3 scripts/compile_paper.py` (requires
pdfLaTeX, BibTeX, PyMuPDF, NumPy, and Matplotlib). The builder regenerates the
audited vector figures, uses a clean temporary auxiliary directory, and writes
the ignored `main.pdf`. Numerical sources, unresolved evidence gaps, and the
validation-selection contracts are documented in `docs/MANUSCRIPT_AUDIT.md`.

## Structure

```text
main.tex                 IEEE conference paper entry point
content/chapters/        section files included by main.tex
content/images/          paper figures
content/tables/          paper tables
content/references.bib   bibliography database
IEEEtran.cls             official IEEE conference document class
IEEEtran.bst             official unsorted IEEE bibliography style
IEEEtranS.bst            official sorted IEEE bibliography style
IEEEabrv.bib             official abbreviated IEEE journal-name strings
IEEEfull.bib             official full journal-name strings
docs/                    IEEE template/BST documentation and example assets
```

The prior LNCS/ACCV files remain in place for historical scaffold provenance,
but `main.tex` does not load them. Do not mix LNCS commands such as `\inst`,
`\institute`, `\authorrunning`, or `\keywords` into this IEEE manuscript.
The working draft intentionally leaves `\author{}` empty because verified
author metadata is not stored here. Before submission, follow the conference's
then-current author-identification instructions and populate IEEE's
`\IEEEauthorblockN`/`\IEEEauthorblockA` structure; do not infer author details.

## RoboticCC 2027 submission constraints

The official 2027 submission page specifies:

- regular manuscripts: **8 pages total**;
- IEEE double-column format on **US letter** paper;
- the limit includes **all figures, tables, and references**;
- up to two extra pages are permitted at **USD 150 per page**; and
- electronic submission is PDF-only through EasyChair.

The default target for this repository is therefore an eight-page regular
paper, including references. Do not interpret an earlier 12-page ACCV writing
brief as applicable to this venue, and do not move excess technical material
into an uncounted appendix: the published limit includes all paper content.
Treat a nine- or ten-page version only as an explicitly approved paid-overlength
fallback.

Official sources:

- <https://www.roboticcomputing.org/submission>
- <https://www.roboticcomputing.org/>

The research-writing overhaul brief is maintained in
`docs/ONLINE_PAPER_OVERHAUL_PROMPT.md` and targets this branch only. The ACCV
manuscript is a read-only source and must not be edited.

## Official IEEE template provenance

- `IEEEtran.cls`, `docs/IEEEtran_HOWTO.pdf`,
  `docs/IEEE-conference-template-062824.tex`,
  `docs/IEEE-conference-template-062824.pdf`, and `docs/fig1.png` came
  unmodified from IEEE's conference-LaTeX template archive.
- `IEEEtran.bst`, `IEEEtranS.bst`, `IEEEabrv.bib`, `IEEEfull.bib`,
  `docs/IEEEexample.bib`, `docs/IEEEtranBST2_README.txt`, and
  `docs/IEEEtran_bst_HOWTO.pdf` came unmodified from IEEEtranBST2.
- `main.tex` is the intentional adaptation: it preserves this repository's
  chapter/image/table/reference structure while using the official IEEE
  conference document and bibliography format.

## Compile

With pdfLaTeX/BibTeX:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The working manuscript enables the bibliography and retains seventeen cited
records in `content/references.bib`. The supported IEEE BST author control
abbreviates long author lists without changing the official bibliography style.

Generated PDFs and auxiliary files should not be committed. Rebuild `main.pdf`
locally after every manuscript change and inspect the complete rendered output.
