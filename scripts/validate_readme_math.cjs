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
require(path.join(dependency, 'input/tex/AllPackages.js'));
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const document = mathjax.document('', {
  InputJax: new TeX({packages: ['base', 'ams'], tags: 'ams', formatError: (_jax, error) => {throw error;}}),
  OutputJax: new SVG({fontCache: 'none'}),
});
const markdown = fs.readFileSync(path.join(root, 'README.md'), 'utf8');
// MathJax alone does not reproduce GitHub's server-side macro filter.
const blocked = /\\(operatorname|DeclareMathOperator|newcommand|renewcommand|def)\b/g;
const filtered = [...markdown.matchAll(blocked)].map(match => match[1]);
if (filtered.length) throw new Error(`Filtered math macros: ${filtered.join(', ')}`);
const blocks = [...markdown.matchAll(/```math\r?\n([\s\S]*?)\r?\n```/g)];
const inline = [...markdown.matchAll(/\$`([^`\r\n]+)`\$/g)];
const formulas = [...blocks.map(m => ({tex:m[1], display:true})),
                  ...inline.map(m => ({tex:m[1], display:false}))];
const rendered = formulas.map((formula, index) => {
  const svg = adaptor.outerHTML(document.convert(formula.tex, {display:formula.display}));
  if (svg.includes('data-mjx-error') || !svg.includes('<svg')) {
    throw new Error(`Formula ${index+1} failed: ${formula.tex}`);
  }
  return `<section><h2>${formula.display ? 'Block' : 'Inline'} ${index+1}</h2>${svg}</section>`;
});
const tags = [...markdown.matchAll(/\\tag\{(\d+)\}/g)].map(m => Number(m[1]));
if (blocks.length !== 47 || tags.some((tag, index) => tag !== index+1)) {
  throw new Error('Expected 47 consecutive numbered block formulas');
}
console.log(JSON.stringify({status:'PASS', numbered_blocks:blocks.length, inline_formulas:inline.length,
                           rendered_svg:rendered.length}, null, 2));
