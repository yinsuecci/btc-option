// Optional isolated MathJax check: install mathjax-full into .math-check first.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const dependency = path.join(root, '.math-check/node_modules/mathjax-full/js');
const {mathjax} = require(path.join(dependency, 'mathjax.js'));
const {TeX} = require(path.join(dependency, 'input/tex.js'));
const {SVG} = require(path.join(dependency, 'output/svg.js'));
const {liteAdaptor} = require(path.join(dependency, 'adaptors/liteAdaptor.js'));
const {RegisterHTMLHandler} = require(path.join(dependency, 'handlers/html.js'));
const {STATE} = require(path.join(dependency, 'core/MathItem.js'));
const {SerializedMmlVisitor} = require(path.join(dependency, 'core/MmlTree/SerializedMmlVisitor.js'));
require(path.join(dependency, 'input/tex/AllPackages.js'));
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const document = mathjax.document('', {
  InputJax: new TeX({packages: ['base', 'ams'], tags: 'none', formatError: (_jax, error) => {throw error;}}),
  OutputJax: new SVG({fontCache: 'none'}),
});
const markdown = fs.readFileSync(path.join(root, 'README.md'), 'utf8');
// MathJax alone does not reproduce GitHub's server-side macro filter.
const blocked = /\\(operatorname|DeclareMathOperator|newcommand|renewcommand|def|tag)\b/g;
const filtered = [...markdown.matchAll(blocked)].map(match => match[1]);
if (filtered.length) throw new Error(`Filtered math macros: ${filtered.join(', ')}`);
const blocks = [...markdown.matchAll(/```math\r?\n([\s\S]*?)\r?\n```/g)];
const inline = [...markdown.matchAll(/\$`([^`\r\n]+)`\$/g)];
const formulas = [...blocks.map(m => ({tex:m[1], display:true})),
                  ...inline.map(m => ({tex:m[1], display:false}))];
const visitor = new SerializedMmlVisitor();
const rendered = formulas.map((formula, index) => {
  if (/[<>]/.test(formula.tex)) throw new Error(`HTML-sensitive relation in formula ${index+1}`);
  const tree = document.convert(formula.tex, {display:formula.display, end:STATE.COMPILED});
  const mathml = visitor.visitTree(tree);
  if (mathml.includes('<merror') || mathml.includes('<mlabeledtr') || !mathml.includes('<math')) {
    throw new Error(`Formula ${index+1} failed: ${formula.tex}`);
  }
  return mathml;
});
const tags = [...markdown.matchAll(/^\*\*式 \((\d+)\)\*\*$/gm)].map(m => Number(m[1]));
if (blocks.length !== 47 || tags.length !== 47 || tags.some((tag, index) => tag !== index+1)) {
  throw new Error('Expected 47 consecutive numbered block formulas');
}
if (process.argv.includes('--preview')) {
  // Native MathML preview catches layout defects hidden by SVG-only validation.
  const sections = rendered.slice(0, blocks.length).map((math, i) =>
    `<section id="equation-${i+1}"><h2>Equation (${i+1})</h2>${math}</section>`);
  fs.writeFileSync(path.join(root, '.math-check/native-math-preview.html'),
    `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native MathML layout check</title><style>body{max-width:760px;margin:24px auto;padding:0 20px;font:18px Georgia,serif;color:#152333;background:#fff}section{border-bottom:1px solid #d8e3ee;padding:16px 0}h2{font:14px Georgia,serif;color:#63788e}math{overflow-x:auto;padding:12px 0}</style><body>${sections.join('')}</body></html>`);
}
console.log(JSON.stringify({status:'PASS', numbered_blocks:blocks.length, inline_formulas:inline.length,
                           native_mathml:rendered.length}, null, 2));
