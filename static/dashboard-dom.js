/* Keyed in-place DOM patching for live-refreshed panels. Renderers keep
 * producing HTML strings; patchHTML reconciles the existing subtree against
 * that markup so nodes keep their identity (keyboard focus, open inputs,
 * chart trigger buttons) and CSS transitions run instead of a full repaint.
 * Elements carrying data-key are matched by key among their siblings; other
 * nodes are matched positionally by node type and tag. */
(function(root, factory){
  if(typeof module === 'object' && module.exports) module.exports = factory();
  else root.TokenfoldDom = factory();
})(typeof window === 'undefined' ? globalThis : window, function(){
  'use strict';
  function keyOf(node){
    return node.nodeType === 1 && node.hasAttribute('data-key') ? node.getAttribute('data-key') : null;
  }
  function sameKind(a, b){
    if(a.nodeType !== b.nodeType) return false;
    if(a.nodeType === 1 && a.tagName !== b.tagName) return false;
    return keyOf(a) === keyOf(b);
  }
  function syncAttributes(target, source){
    for(var i = target.attributes.length - 1; i >= 0; i--){
      var name = target.attributes[i].name;
      if(!source.hasAttribute(name)) target.removeAttribute(name);
    }
    for(var j = 0; j < source.attributes.length; j++){
      var attr = source.attributes[j];
      if(target.getAttribute(attr.name) !== attr.value) target.setAttribute(attr.name, attr.value);
    }
  }
  function patch(target, source){
    if(!sameKind(target, source)){ target.parentNode.replaceChild(source, target); return source; }
    if(target.nodeType === 3 || target.nodeType === 8){
      if(target.nodeValue !== source.nodeValue) target.nodeValue = source.nodeValue;
      return target;
    }
    if(target.nodeType !== 1) return target;
    syncAttributes(target, source);
    patchChildren(target, source);
    return target;
  }
  function patchChildren(target, source){
    var wanted = Array.prototype.slice.call(source.childNodes);
    // A Map keeps arbitrary key strings ('__proto__', 'constructor') inert.
    var keyed = new Map();
    Array.prototype.forEach.call(target.childNodes, function(child){
      var key = keyOf(child);
      if(key !== null && !keyed.has(key)) keyed.set(key, child);
    });
    var cursor = target.firstChild;
    wanted.forEach(function(next){
      var key = keyOf(next), match = null;
      if(key !== null){ match = keyed.get(key) || null; keyed.delete(key); }
      else if(cursor && keyOf(cursor) === null && sameKind(cursor, next)) match = cursor;
      if(match){
        if(match !== cursor) target.insertBefore(match, cursor);
        cursor = match.nextSibling;
        patch(match, next);
      } else {
        target.insertBefore(next, cursor);
      }
    });
    while(cursor){ var drop = cursor; cursor = cursor.nextSibling; target.removeChild(drop); }
  }
  function patchHTML(container, html){
    var template = container.ownerDocument.createElement('template');
    template.innerHTML = html;
    patchChildren(container, template.content);
    return container;
  }
  return {patch:patch, patchChildren:patchChildren, patchHTML:patchHTML};
});
