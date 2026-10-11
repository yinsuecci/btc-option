import re
from pathlib import Path


def test_readme_contains_complete_github_native_math():
    root = Path(__file__).resolve().parents[1]
    document = (root / 'README.md').read_text(encoding='utf-8')
    assert '![' not in document and 'docs/math/' not in document
    assert 'option-backtest.pdf' not in document
    assert 'option-backtest.tex' not in document
    blocks = re.findall(r'^```math\n([\s\S]*?)\n```$', document, re.MULTILINE)
    inline = re.findall(r'\$`([^`\n]+)`\$', document)
    assert len(blocks) == 47
    assert len(inline) == 68
    numbers = re.findall(r'^\*\*式 \((\d+)\)\*\*$', document, re.MULTILINE)
    assert [int(n) for n in numbers] == list(range(1, 48))
    paired = re.findall(r'^\*\*式 \((\d+)\)\*\*\n\n```math\n', document, re.MULTILINE)
    assert paired == numbers
    assert not list((root / 'docs').glob('option-backtest.*'))


def test_readme_avoids_filtered_math_macros():
    document = (Path(__file__).resolve().parents[1] / 'README.md').read_text(encoding='utf-8')
    # Full MathJax accepts operatorname, but GitHub's macro filter rejects it.
    blocked = {'operatorname', 'DeclareMathOperator', 'newcommand', 'renewcommand', 'def', 'tag'}
    commands = set(re.findall(r'\\([A-Za-z]+)', document))
    assert not commands.intersection(blocked)


def test_readme_math_avoids_html_truncation_and_numbered_mathml_rows():
    document = (Path(__file__).resolve().parents[1] / 'README.md').read_text(encoding='utf-8')
    blocks = re.findall(r'^```math\n([\s\S]*?)\n```$', document, re.MULTILINE)
    inline = re.findall(r'\$`([^`\n]+)`\$', document)
    assert all('<' not in math and '>' not in math for math in blocks + inline)
    assert r'0\lt P_i\lt1' in blocks[0]
    assert r's\lt t' in blocks[7]
    assert r'a_s\lt d_t' in blocks[7]
    assert all(r'\tag' not in block for block in blocks)
    assert r'\begin{gathered}' in blocks[2]
    assert all(r'\begin{aligned}' not in block for block in blocks)
