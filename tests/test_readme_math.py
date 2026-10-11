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
    assert [int(n) for n in re.findall(r'\\tag\{(\d+)\}', document)] == list(range(1, 48))
    assert not list((root / 'docs').glob('option-backtest.*'))


def test_readme_avoids_filtered_math_macros():
    document = (Path(__file__).resolve().parents[1] / 'README.md').read_text(encoding='utf-8')
    # Full MathJax accepts operatorname, but GitHub's macro filter rejects it.
    blocked = {'operatorname', 'DeclareMathOperator', 'newcommand', 'renewcommand', 'def'}
    commands = set(re.findall(r'\\([A-Za-z]+)', document))
    assert not commands.intersection(blocked)
