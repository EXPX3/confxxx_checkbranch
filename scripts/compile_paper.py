#!/usr/bin/env python3
"""Clean IEEE build and structural checks for the expanded working manuscript."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

import fitz

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'main.pdf')
    parser.add_argument('--expected-body-pages', type=int, default=12)
    parser.add_argument('--max-reference-pages', type=int, default=1)
    args = parser.parse_args()
    subprocess.run(['python3', str(ROOT / 'scripts/build_paper_assets.py')],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    with tempfile.TemporaryDirectory(prefix='roboticcc-paper-') as directory:
        build = Path(directory)
        if shutil.which('pdflatex') and shutil.which('bibtex'):
            latex = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                     f'-output-directory={build}', 'main.tex']
            subprocess.run(latex, cwd=ROOT, check=True, capture_output=True, text=True)
            env = dict(os.environ, BIBINPUTS=f'{ROOT}:', BSTINPUTS=f'{ROOT}:')
            subprocess.run(['bibtex', 'main'], cwd=build, env=env, check=True,
                           capture_output=True, text=True)
            for _ in range(2):
                subprocess.run(latex, cwd=ROOT, check=True,
                               capture_output=True, text=True)
        elif shutil.which('tectonic'):
            subprocess.run(
                ['tectonic', '--keep-logs', '--keep-intermediates',
                 '--outdir', str(build), 'main.tex'], cwd=ROOT, check=True,
                capture_output=True, text=True)
        else:
            raise RuntimeError('Neither pdfLaTeX/BibTeX nor Tectonic is available')
        log = (build / 'main.log').read_text()
        problems = {
            'overfull box': r'Overfull \\[hv]box',
            'undefined reference or citation':
                r'(LaTeX Warning:.*(?:Reference|Citation).*undefined|'
                r'There were undefined references)',
            'multiply defined label': r'multiply defined',
        }
        for label, pattern in problems.items():
            if re.search(pattern, log, re.I):
                raise ValueError(f'Unresolved LaTeX problem: {label}')
        pdf = fitz.open(build / 'main.pdf')
        text = '\n'.join(page.get_text() for page in pdf)
        references = [i + 1 for i, page in enumerate(pdf)
                      if re.search(r'^REFERENCES$', page.get_text(), re.M | re.I)]
        if len(references) != 1:
            raise ValueError(f'Expected one references section, found pages {references}')
        body_pages = references[0] - 1
        reference_pages = len(pdf) - body_pages
        if body_pages != args.expected_body_pages:
            raise ValueError(
                f'Expected {args.expected_body_pages} body pages, found {body_pages}')
        if reference_pages > args.max_reference_pages:
            raise ValueError(
                f'Expected at most {args.max_reference_pages} reference pages, '
                f'found {reference_pages}')
        if any(abs(page.rect.width - 612) > .01 or
               abs(page.rect.height - 792) > .01 for page in pdf):
            raise ValueError('PDF is not US letter')
        forbidden = re.compile(
            r'V07b|V5[- ]V2|v5v2|v5v2dinov3|\bVariant\s+[0-9]|DGX|UAV11|'
            r'altitude[ -]interpolation|taxonomy[ -]remap|'
            r'GT[ -]present|union[ -]present', re.I)
        source = (ROOT / 'main.tex').read_text() + '\n'.join(
            p.read_text() for p in (ROOT / 'content/chapters').glob('*.tex'))
        if forbidden.search(text + source + json.dumps(pdf.metadata)):
            raise ValueError('Internal label, removed analysis, or alternative semantic average found')
        if pdf.metadata.get('author'):
            raise ValueError('Unexpected author metadata')
        report = {
            'page_count': len(pdf), 'body_page_count': body_pages,
            'reference_pages': list(range(references[0], len(pdf) + 1)),
            'page_size_points': [612, 792], 'semantic_average': 'standard fixed 21 occupied classes',
            'other_semantic_averages_in_paper': False,
            'undefined_citations_or_overfull_boxes': False,
            'prohibited_paper_labels_found': False,
            'author_metadata': pdf.metadata.get('author', ''),
            'ieeetran_cls_sha256': hashlib.sha256((ROOT / 'IEEEtran.cls').read_bytes()).hexdigest(),
            'ieeetran_bst_sha256': hashlib.sha256((ROOT / 'IEEEtran.bst').read_bytes()).hexdigest(),
        }
        pdf.close()
        args.output = args.output.resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build / 'main.pdf', args.output)
        (ROOT / 'docs/evidence/layout_validation.json').write_text(
            json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
