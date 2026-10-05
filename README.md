# IEEE RCC 2027 Paper Scaffold

This checkout is the IEEE RCC 2027 adaptation of the reusable conference-paper
scaffold. It was created from `main` of `EXPX3/confxxx_checkbranch` on the local
branch `ieeercc2027`. The existing `content/` layout is deliberately retained.

The root `main.tex` now uses the official IEEE conference class and bibliography
style. The unmodified official source files from the supplied IEEE archives are
included locally so the manuscript can be compiled reproducibly without relying
on a system-installed version.

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
Use IEEE's `\IEEEauthorblockN`, `\IEEEauthorblockA`, and
`\begin{IEEEkeywords}...\end{IEEEkeywords}` constructs instead.

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

Generated PDFs and auxiliary files should not be committed.
