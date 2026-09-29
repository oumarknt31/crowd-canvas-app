// Exercise the component's event handlers and Streamlit message protocol.
// This checks logic with SVG-coordinate stubs, not a real browser rendering.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const events = [], handlers = {}, nodes = {};
function node(id) {
  return nodes[id] ||= {style: {}, attributes: {}, textContent: '',
    setAttribute(k, v) { this.attributes[k] = v; },
    addEventListener(k, fn) { handlers[k] = fn; },
    setPointerCapture() {}, getScreenCTM() { return {inverse() { return {}; }}; },
    createSVGPoint() { return {x: 0, y: 0, matrixTransform() { return this; }}; }};
}
const source = fs.readFileSync('components/light_compass/index.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
vm.runInNewContext(source, {
  document: {querySelector() { return node('svg'); }, getElementById: node, body: node('body')},
  window: {parent: {postMessage(data) { events.push(data); }},
    addEventListener(type, fn) { handlers[type] = fn; }},
});
handlers.message({data: {type: 'streamlit:render', args: {angle: 225}}});
handlers.keydown({key: 'ArrowRight', preventDefault() {}});
assert.equal(events.filter(e => e.type === 'streamlit:setComponentValue').at(-1).value, 230);
handlers.pointerdown({pointerId: 1, clientX: 110, clientY: 162});
handlers.pointerup({pointerId: 1, clientX: 110, clientY: 162});
assert.equal(events.filter(e => e.type === 'streamlit:setComponentValue').at(-1).value, 90);
assert(events.some(e => e.type === 'streamlit:componentReady' && e.apiVersion === 1));
console.log('Sun compass keyboard, drag-angle calculation and Streamlit messages passed.');
