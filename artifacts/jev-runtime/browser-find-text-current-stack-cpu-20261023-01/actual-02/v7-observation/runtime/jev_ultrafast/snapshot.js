(() => {
  if (!document.body) return null;
  const cache = window.__jevFast ||= {ids:new WeakMap(), nodes:new Map(), next:1};
  const identity = e => {
    if (!cache.ids.has(e)) cache.ids.set(e,cache.next++);
    const id=cache.ids.get(e); cache.nodes.set(id,e); return id;
  };
  for (const [id,e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);
  const safe = e => !['password','file','hidden'].includes(e.type);
  const visible = e => !e.closest('[aria-hidden="true"],[inert]') &&
    e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  const name = (e,seen=new Set()) => {
    if (!e || seen.has(e)) return '';
    seen.add(e);
    const referenced=(e.getAttribute('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(document.getElementById(id),seen)).filter(Boolean).join(' ');
    return referenced || e.getAttribute('aria-label') ||
      [...(e.labels||[])].map(l=>name(l,seen)).filter(Boolean).join(' ') ||
      (['button','submit','reset'].includes(e.type) ? e.value : '') || e.getAttribute('alt') ||
      (e.tagName==='INPUT' ? '' : [...e.childNodes].map(n=>n.nodeType===3 ? n.textContent :
        n.nodeType===1 && n.getAttribute('aria-hidden')!=='true' ? name(n,seen) : '').join(' ').trim()) ||
      e.getAttribute('title') || e.getAttribute('placeholder') || '';
  };
  const roles=['button','link','checkbox','radio','switch','tab','menuitem','menuitemradio',
    'option','gridcell','combobox','textbox','searchbox','spinbutton'];
  const selector='a[href],button,input,textarea,select,summary,[contenteditable="true"],'+
    roles.map(role=>'[role="'+role+'"]').join(',');
  const role = e => {
    const explicit=e.getAttribute('role');
    if (roles.includes(explicit)) return explicit;
    if (e.tagName==='BUTTON' || e.tagName==='SUMMARY') return 'button';
    if (e.tagName==='A') return 'link';
    if (e.tagName==='SELECT') return 'combobox';
    if (e.tagName==='TEXTAREA' || e.isContentEditable) return 'textbox';
    if (e.tagName==='INPUT') {
      if (['checkbox','radio'].includes(e.type)) return e.type;
      if (['button','submit','reset','image'].includes(e.type)) return 'button';
      if (e.type==='search') return 'searchbox';
      if (e.type==='number') return 'spinbutton';
      if (['text','email','url','tel'].includes(e.type)) return 'textbox';
    }
    return null;
  };
  cache.pageKey=()=>[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    [...document.querySelectorAll('input,textarea,select')].filter(safe)
      .map(e=>[identity(e),e.value,e.checked,e.selectedIndex,e.disabled,e.readOnly])];
  cache.guard=e=>{
    if (!e?.isConnected || !visible(e)) return null;
    const scope=e.closest('form,dialog,[role="dialog"],article,li,tr,[role="row"]') || e.parentElement;
    return [identity(e),role(e),name(e),e.value??null,e.checked??null,e.selectedIndex??null,
      e.readOnly??null,e.matches(':disabled'),e.getAttribute('aria-disabled'),
      e.getAttribute('aria-expanded'),e.getAttribute('aria-checked'),e.getAttribute('aria-selected'),
      e.getAttribute('href'),scope?.innerText?.slice(0,6000)||''];
  };
  const actions=[];
  for (const e of document.querySelectorAll(selector)) {
    if (!safe(e) || !visible(e) || e.matches(':disabled') || e.closest('[aria-disabled="true"]')) continue;
    const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2, rname=role(e);
    if (!rname || r.width<=0 || r.height<=0 || x<0 || y<0 || x>=innerWidth || y>=innerHeight) continue;
    if (rname==='gridcell' && e.querySelector('button,[role="button"]')) continue;
    const base={node:identity(e),role:rname,label:name(e)||rname,
      rect:{x:r.x,y:r.y,w:r.width,h:r.height}};
    for (const key of ['checked','selected','expanded']) {
      const value=e.getAttribute('aria-'+key);
      if (value!==null) base[key]=value;
    }
    if (['checkbox','radio'].includes(e.type)) base.checked=String(e.checked);
    if (e.tagName==='SELECT') {
      for (const o of e.options) if (!o.selected && !o.disabled && !o.closest('optgroup[disabled]'))
        actions.push({...base,kind:'select',value:o.value,
          current_value:[...e.selectedOptions].map(o=>o.label).join(', '),label:base.label+' → '+o.label});
    } else {
      const editable=!e.readOnly && e.getAttribute('aria-readonly')!=='true' &&
        (['textbox','searchbox','spinbutton'].includes(rname) ||
          (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
      const value='value' in e ? String(e.value) :
        e.isContentEditable || rname==='combobox' ? e.innerText.trim() : '';
      actions.push({...base,kind:editable?'fill':'click',value});
      if (editable) actions.push({...base,kind:'click',value,label:'Open '+base.label});
    }
  }
  const words=[], range=document.createRange(); let length=0;
  const textProfile={id:'composed-visible-text-settled-v7',visited_nodes:0,
    node_limit:20000,depth_limit:128,node_limit_reached:false,depth_limit_reached:false};
  const composedParent=e=>e?.assignedSlot || e?.parentElement || e?.getRootNode?.()?.host || null;
  const excludedElement=e=>{
    const style=getComputedStyle(e);
    return e.hidden || e.hasAttribute('inert') ||
      e.getAttribute('aria-hidden')==='true' || style.display==='none' ||
      style.visibility==='hidden' || style.visibility==='collapse' ||
      Number(style.opacity)===0 || style.contentVisibility==='hidden';
  };
  const composedVisible=e=>{
    let depth=0;
    for (let current=e; current; current=composedParent(current)) {
      if (++depth>128) {textProfile.depth_limit_reached=true;return false;}
      if (excludedElement(current)) return false;
    }
    return true;
  };
  const seen=new WeakSet(), stack=[{node:document.body,depth:0}];
  while (stack.length && length<6000) {
    const {node,depth}=stack.pop();
    if (seen.has(node)) continue;
    seen.add(node);
    if (textProfile.visited_nodes>=textProfile.node_limit) {
      textProfile.node_limit_reached=true;break;
    }
    textProfile.visited_nodes++;
    if (depth>textProfile.depth_limit) {textProfile.depth_limit_reached=true;continue;}
    if (node.nodeType===Node.TEXT_NODE) {
      const value=node.textContent.trim(), parent=node.assignedSlot || node.parentElement || node.getRootNode()?.host;
      if (!value || !parent || parent.closest('script,style,noscript,template') ||
          !composedVisible(parent)) continue;
      range.selectNodeContents(node);const r=range.getBoundingClientRect();
      if (r.width>0 && r.height>0 && r.bottom>0 && r.top<innerHeight &&
          r.right>0 && r.left<innerWidth) {
        words.push(value);length+=value.length;
      }
      continue;
    }
    if (node.nodeType===Node.ELEMENT_NODE &&
        node.matches('script,style,noscript,template')) continue;
    if (node.nodeType===Node.ELEMENT_NODE && excludedElement(node)) continue;
    let children;
    if (node.nodeType===Node.ELEMENT_NODE && node.tagName==='SLOT') {
      const assigned=node.assignedNodes({flatten:true});
      children=assigned.length ? assigned : [...node.childNodes];
    } else if (node.nodeType===Node.ELEMENT_NODE && node.shadowRoot?.mode==='open') {
      children=[...node.shadowRoot.childNodes];
    } else children=[...node.childNodes];
    children=children.filter(child=>child.nodeType!==Node.COMMENT_NODE &&
      !(child.nodeType===Node.TEXT_NODE && !child.textContent.trim()));
    for (let i=children.length-1;i>=0;i--) stack.push({node:children[i],depth:depth+1});
  }
  if (textProfile.node_limit_reached || textProfile.depth_limit_reached) {
    const error=new Error('COMPOSED_VISIBLE_TEXT_LIMIT '+JSON.stringify(textProfile));
    error.name='ComposedVisibleTextLimitError';throw error;
  }
  const text=words.join('\n').slice(0,6000), height=document.documentElement.scrollHeight;
  const page_key=cache.pageKey(), guards={};
  for (const a of actions) if (!(a.node in guards)) guards[a.node]=cache.guard(cache.nodes.get(a.node));
  // Compare meaning and identity. Geometry is always resolved and hit-tested just before input.
  const semantics=actions.map(({rect,...action})=>action);
  const marker=[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    document.title,text,semantics,page_key[6]];
  const omitted_actions=Math.max(0,actions.length-250);
  actions.splice(250);
  actions.forEach((a,i)=>a.id='e'+(i+1));
  if (scrollY+innerHeight<height-2) actions.push({id:'scroll_down',kind:'scroll',label:'Scroll down',delta:560});
  if (scrollY>0) actions.push({id:'scroll_up',kind:'scroll',label:'Scroll up',delta:-560});
  actions.push({id:'wait',kind:'wait',label:'Wait for the page to update'});
  return {observation_profile:textProfile,url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,omitted_actions};
})()
