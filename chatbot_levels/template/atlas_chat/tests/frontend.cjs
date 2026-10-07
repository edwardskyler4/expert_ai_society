// Dependency-free checks for untrusted Markdown and rendering structure.
// These do not replace testing in a real browser.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Node {
  constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.value='';this.classList={toggle(){},add(){},remove(){},contains(){return false;}};}
  addEventListener(name,callback){this.listeners[name]=callback;}
  append(...children){this.children.push(...children);}
  setAttribute(name,value){this[name]=value;}
  get textContent(){return this._text??this.children.map(child=>child.textContent||'').join('');}
  set textContent(value){this._text=String(value);this.children=[];}
}
const nodes=new Map();
const document={getElementById(id){if(!nodes.has(id))nodes.set(id,new Node());return nodes.get(id);},querySelectorAll(){return [];},addEventListener(){},createElement(tag){return new Node(tag);},createDocumentFragment(){return new Node('fragment');}};
const context=vm.createContext({document,URL,localStorage:{getItem(){return null;},setItem(){}},setTimeout,clearTimeout,console});
const code=fs.readFileSync(path.join(__dirname,'..','static','app.js'),'utf8').replace(/\ninit\(\);\s*$/,'');
vm.runInContext(code,context);
const functions=vm.runInContext('({inline, markdown, safeURL, escapeHTML})',context);
const flatten=node=>(node.innerHTML||node._text||'')+node.children.map(flatten).join('');

test('inline rendering escapes active HTML and rejects executable URLs',()=>{
  const value=functions.inline('<img src=x onerror=alert(1)> **bold** [unsafe](javascript:alert(1))');
  assert.match(value,/&lt;img/);assert.match(value,/<strong>bold<\/strong>/);assert.doesNotMatch(value,/<img|href="javascript:/);
});
test('links preserve readable labels and use only HTTP(S)',()=>{
  assert.equal(functions.safeURL('data:text/html,evil'),null);
  assert.equal(functions.safeURL('javascript:alert(1)'),null);
  assert.match(functions.inline('[Docs](https://example.com/docs)'),/rel="noopener noreferrer"/);
});
test('fenced code is kept as text and has a copy control',()=>{
  const rendered=functions.markdown('```html\n<script>alert(1)</script>\n```');
  const block=rendered.children[0];assert.equal(block.className,'code-block');
  const pre=block.children[1];assert.equal(pre.tag,'pre');assert.equal(pre.children[0].textContent,'<script>alert(1)</script>');
  assert.equal(pre.children[0].innerHTML,undefined);
});
test('Markdown tables and lists have semantic structure',()=>{
  const rendered=functions.markdown('| Name | Value |\n| --- | --- |\n| alpha | 1 |\n\n- first\n- second');
  assert.equal(rendered.children[0].tag,'table');assert.equal(rendered.children[1].tag,'ul');
  assert.equal(rendered.children[1].children.length,2);
});
test('malformed and unknown Markdown remains harmless text',()=>{
  const rendered=functions.markdown('# Heading\n\n<img onerror="evil">\n\n[bad](data:text/html,evil)');
  const output=flatten(rendered);assert.doesNotMatch(output,/<img|href="data:/);assert.match(output,/&lt;img/);
});
