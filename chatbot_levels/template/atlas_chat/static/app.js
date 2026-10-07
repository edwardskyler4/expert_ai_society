'use strict';
const $ = (id) => document.getElementById(id);
const state = {config:null, conversations:[], current:null, busy:false, runId:null, cancelRequested:false, attachments:[], documents:[], memories:[], pinned:true};
const defaultPrefs = {provider:'ollama', model:'', max_tokens:4096, temperature:0.7, reasoning:'', system:'', documents:true, web:false, code:false};
function saved(key, fallback) {try {return JSON.parse(localStorage.getItem(key)) ?? fallback;} catch {return fallback;}}
function persist(key,value) {try {localStorage.setItem(key,JSON.stringify(value));} catch {}}
let prefs = {...defaultPrefs, ...saved('atlas.preferences',{})};
let toastTimer;
function toast(message) {$('toast').textContent=message; $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,5000);}
function element(tag, cls, text) {const node=document.createElement(tag); if(cls)node.className=cls; if(text!==undefined)node.textContent=text; return node;}
async function api(path, options={}) {
  const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...options.headers}});
  let value; try {value=await response.json();} catch {throw new Error('The server returned an unreadable response.');}
  if(!response.ok)throw new Error(value.error||`Request failed (${response.status}).`);
  return value;
}
const post=(path,body)=>api(path,{method:'POST',body:JSON.stringify(body)});
const remove=(path)=>api(path,{method:'DELETE'});
function escapeHTML(text) {return String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function safeURL(value) {try {const u=new URL(value); return ['http:','https:'].includes(u.protocol)?u.href:null;}catch{return null;}}
function inline(text) {
  const pieces=[];
  // Protect inline code and links before escaping the remaining text.
  let raw=String(text).replace(/`([^`]+)`|\[([^\]]+)\]\(([^\s)]+)\)/g,(match,code,label,url)=>{
    let html;
    if(code!==undefined)html=`<code>${escapeHTML(code)}</code>`;
    else {const href=safeURL(url);html=href?`<a href="${escapeHTML(href)}" target="_blank" rel="noopener noreferrer">${escapeHTML(label)}</a>`:escapeHTML(match);}
    const token=`\u0000${pieces.length}\u0000`;pieces.push(html);return token;
  });
  raw=escapeHTML(raw).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\*([^*\n]+)\*/g,'<em>$1</em>');
  return raw.replace(/\u0000(\d+)\u0000/g,(_,index)=>pieces[Number(index)]||'');
}
function markdown(text) {
  const root=document.createDocumentFragment(), lines=String(text).split('\n');let i=0;
  const cells=line=>line.replace(/^\s*\||\|\s*$/g,'').split('|').map(cell=>cell.trim());
  while(i<lines.length) {
    const line=lines[i];
    if(!line.trim()){i++;continue;}
    if(/^\s*```/.test(line)) {
      const language=line.trim().slice(3).trim(), content=[];i++;
      while(i<lines.length&&!/^\s*```/.test(lines[i]))content.push(lines[i++]);if(i<lines.length)i++;
      const block=element('div','code-block'), header=element('div','code-header'), copy=element('button','', 'Copy');copy.type='button';
      header.append(element('span','',language||'code'),copy);const pre=document.createElement('pre'), code=element('code','',content.join('\n'));pre.append(code);
      copy.addEventListener('click',()=>copyText(code.textContent));block.append(header,pre);root.append(block);continue;
    }
    if(i+1<lines.length&&line.includes('|')&&/^\s*\|?\s*:?-{3,}/.test(lines[i+1])) {
      const table=document.createElement('table'), head=document.createElement('thead'), tr=document.createElement('tr');
      for(const value of cells(line)){const th=document.createElement('th');th.innerHTML=inline(value);tr.append(th);}head.append(tr);table.append(head);i+=2;
      const body=document.createElement('tbody');while(i<lines.length&&lines[i].includes('|')&&lines[i].trim()){const row=document.createElement('tr');for(const value of cells(lines[i++])){const td=document.createElement('td');td.innerHTML=inline(value);row.append(td);}body.append(row);}table.append(body);root.append(table);continue;
    }
    const heading=line.match(/^(#{1,3})\s+(.+)$/);
    if(heading){const h=document.createElement('h'+heading[1].length);h.innerHTML=inline(heading[2]);root.append(h);i++;continue;}
    if(/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const ordered=/^\s*\d+\./.test(line), list=document.createElement(ordered?'ol':'ul');
      const pattern=ordered?/^\s*\d+\.\s+/:/^\s*[-*]\s+/;
      while(i<lines.length&&pattern.test(lines[i])){const li=document.createElement('li');li.innerHTML=inline(lines[i++].replace(pattern,''));list.append(li);}root.append(list);continue;
    }
    if(/^>\s?/.test(line)){const quote=document.createElement('blockquote');quote.innerHTML=inline(line.replace(/^>\s?/,''));root.append(quote);i++;continue;}
    const paragraph=[];do{paragraph.push(lines[i++]);}while(i<lines.length&&lines[i].trim()&&!/^(#{1,3}\s|\s*```|\s*[-*]\s|\s*\d+\.\s|>\s?)/.test(lines[i]));
    const p=document.createElement('p');p.innerHTML=paragraph.map(inline).join('<br>');root.append(p);
  }
  return root;
}
async function copyText(text){try{await navigator.clipboard.writeText(text);toast('Copied to clipboard.');}catch{toast('Clipboard access is unavailable in this browser.');}}
function scrollDown(){if(state.pinned)$('scroll-area').scrollTop=$('scroll-area').scrollHeight;}
$('scroll-area').addEventListener('scroll',()=>{const el=$('scroll-area');state.pinned=el.scrollHeight-el.scrollTop-el.clientHeight<100;});
function updatePrefsUI(){
  const provider=prefs.provider==='openai'?'Hosted':prefs.provider==='demo'?'Demo':'Local';
  $('model-label').textContent=(prefs.model||'Select model')+' · '+provider;
  $('connection-label').textContent=prefs.provider==='demo'?'Interface demo · no AI':prefs.provider==='openai'?(state.config?.openai_configured?'API key configured':'API key not configured'):'Ollama · connection not checked';
  $('demo-banner').hidden=prefs.provider!=='demo';
  for(const name of ['documents','web','code']){const button=$(name+'-toggle');button.classList.toggle('active',!!prefs[name]);button.setAttribute('aria-pressed',String(!!prefs[name]));if(name!=='documents')button.title=prefs.provider==='openai'?'Allow the model to use this hosted tool':'Requires the OpenAI provider';}
}
function renderConversations(){
  const nav=$('conversations'), filter=$('chat-search').value.toLowerCase();nav.replaceChildren();
  const rows=state.conversations.filter(item=>item.title.toLowerCase().includes(filter));
  if(!rows.length){nav.append(element('div','empty-nav',filter?'No matching conversations.':'Your conversations will appear here.\nStart with a question or an idea.'));return;}
  for(const item of rows){const row=element('div','conversation-row'+(state.current?.id===item.id?' active':''));const select=element('button','conversation-select');select.title=item.title;select.append(element('span','chat-icon','▱'),element('span','chat-name',item.title));select.addEventListener('click',()=>openConversation(item.id));const menu=element('button','conversation-menu','⋯');menu.setAttribute('aria-label','Manage '+item.title);menu.addEventListener('click',()=>manageConversation(item));row.append(select,menu);nav.append(row);}
}
async function refreshConversations(){state.conversations=await api('/api/conversations');renderConversations();}
async function newConversation(){if(state.busy){toast('Stop or finish the current reply first.');return;}state.current=null;state.attachments=[];renderAttachments();renderMessages();renderConversations();$('sidebar').classList.remove('open');$('prompt').focus();}
async function openConversation(id){if(state.busy){toast('Stop or finish the current reply first.');return;}try{state.current=await api('/api/conversations/'+id);state.attachments=[];state.pinned=true;renderAttachments();renderConversations();renderMessages();$('sidebar').classList.remove('open');scrollDown();}catch(error){toast(error.message);}}
function renderMessages(){
  const messages=state.current?.messages||[];$('messages').replaceChildren();$('welcome').hidden=messages.length>0;$('chat-title').textContent=state.current?.title||'New conversation';
  for(const message of messages)$('messages').append(messageNode(message));scrollDown();
}
function showSource(source){$('source-title').textContent=source.title;$('source-location').textContent=`${source.label} · passage ${source.chunk}`;$('source-text').textContent=source.text;$('source-dialog').showModal();}
function messageNode(message){
  const root=element('article','message '+message.role);root.id='message-'+message.id;
  root.append(element('div','message-avatar',message.role==='user'?'YOU':'✳'));
  const body=element('div','message-body'),label=element('div','message-label',message.role==='user'?'You':'Atlas');
  if(message.role==='assistant'&&message.metadata?.model)label.append(element('span','',message.metadata.model));body.append(label);
  if(message.images?.length){const images=element('div','message-images');for(const id of message.images){const image=document.createElement('img');image.src='/api/images/'+id;image.alt='Attached image';image.loading='lazy';images.append(image);}body.append(images);}
  const content=element('div','message-content'+(message.status==='streaming'?' streaming':''));content.dataset.content='true';
  if(message.role==='assistant'&&message.status!=='streaming')content.append(markdown(message.content));else content.textContent=message.content;body.append(content);
  const metadata=message.metadata||{};
  if(metadata.sources?.length){const sources=element('div','source-list');for(const source of metadata.sources){if(source.kind==='web'){const url=safeURL(source.url);if(!url)continue;const a=element('a','source-chip','◎ '+source.title);a.href=url;a.target='_blank';a.rel='noopener noreferrer';sources.append(a);}else{const button=element('button','source-chip',source.label+' · '+source.title);button.title='Inspect retrieved passage';button.addEventListener('click',()=>showSource(source));sources.append(button);}}body.append(sources);}
  if(metadata.artifacts?.length){const files=element('div','source-list');metadata.artifacts.forEach((file,index)=>{const link=element('a','source-chip','↓ '+file.name);link.href=`/api/artifact?conversation=${encodeURIComponent(state.current.id)}&message=${encodeURIComponent(message.id)}&index=${index}`;files.append(link);});body.append(files);}
  for(const warning of metadata.warnings||[])body.append(element('div','message-warning',warning));
  if(metadata.error)body.append(element('div','message-error',metadata.error));
  if(message.status==='interrupted')body.append(element('div','message-warning','The server restarted before this reply finished.'));
  if(message.status==='cancelled')body.append(element('div','message-warning','Generation stopped.'));
  if(message.status!=='streaming'&&message.role==='assistant'){
    const actions=element('div','message-actions'),copy=element('button','','Copy');copy.addEventListener('click',()=>copyText(message.content));actions.append(copy);
    const last=state.current?.messages.at(-1);if(last?.id===message.id){const retry=element('button','','Ask again');retry.addEventListener('click',()=>{const user=[...state.current.messages].reverse().find(item=>item.role==='user');if(user)send(user.content,user.images);});actions.append(retry);}
    if(metadata.elapsed_seconds!==undefined)actions.append(element('span','',metadata.elapsed_seconds+'s'));
    const usage=metadata.usage||{};if(usage.output_tokens!==undefined)actions.append(element('span','',usage.output_tokens.toLocaleString()+' output tokens'));
    body.append(actions);
  }
  root.append(body);return root;
}
function setBusy(value){state.busy=value;$('send-button').hidden=value;$('stop-button').hidden=!value;$('prompt').disabled=value;$('attach-button').disabled=value;$('new-chat').disabled=value;if(!value){$('stream-status').textContent='';state.runId=null;state.cancelRequested=false;}}
async function stop(){if(!state.busy)return;state.cancelRequested=true;$('stream-status').textContent='Stopping after the provider yields its next event…';if(state.runId){try{await post('/api/cancel',{run_id:state.runId});}catch(error){toast(error.message);}}}
async function send(explicitText, explicitImages){
  if(state.busy)return;
  const text=typeof explicitText==='string'?explicitText:$('prompt').value.trim();if(!text)return;
  const images=explicitImages||state.attachments.map(item=>item.id);setBusy(true);state.pinned=true;
  let assistant=null, done=false, accepted=false;
  try{
    if(!state.current){state.current=await post('/api/conversations',{});await refreshConversations();}
    const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversation_id:state.current.id,text,images,options:prefs})});
    if(!response.ok){const error=await response.json();throw new Error(error.error||'Unable to start the response.');}
    accepted=true;$('prompt').value='';resizePrompt();state.attachments=[];renderAttachments();
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    const event=async value=>{
      if(value.type==='start'){
        state.runId=value.run_id;if(state.cancelRequested)await post('/api/cancel',{run_id:state.runId});
        const user={id:value.user_id,role:'user',content:text,status:'complete',images,metadata:{}};
        assistant={id:value.assistant_id,role:'assistant',content:'',status:'streaming',metadata:{model:prefs.model}};
        state.current.messages.push(user,assistant);if(state.current.title==='New conversation')state.current.title=text.replace(/\n/g,' ').slice(0,60);renderMessages();
      }else if(value.type==='delta'&&assistant){assistant.content+=value.text;const content=$('message-'+assistant.id)?.querySelector('[data-content]');if(content)content.textContent=assistant.content;scrollDown();
      }else if(value.type==='status'&&!state.cancelRequested){$('stream-status').textContent=value.text;
      }else if(value.type==='context'&&assistant){assistant.metadata.sources=value.sources;assistant.metadata.warnings=value.warnings;
      }else if(value.type==='error'){toast(value.message);
      }else if(value.type==='done'&&assistant){done=true;assistant.content=value.content;assistant.status=value.status;assistant.metadata=value.metadata;$('message-'+assistant.id)?.replaceWith(messageNode(assistant));scrollDown();}
    };
    while(true){const {value,done:ended}=await reader.read();buffer+=ended?decoder.decode():decoder.decode(value,{stream:true});buffer=buffer.replace(/\r\n/g,'\n');let boundary;while((boundary=buffer.indexOf('\n\n'))>=0){const block=buffer.slice(0,boundary);buffer=buffer.slice(boundary+2);const data=block.split('\n').filter(line=>line.startsWith('data:')).map(line=>line.slice(5).trimStart()).join('\n');if(data)await event(JSON.parse(data));}if(ended)break;}
    if(!done)throw new Error('The stream disconnected. Reload the conversation to see its saved state.');
  }catch(error){toast(error.message);if(!accepted&&typeof explicitText!=='string')$('prompt').value=text;
  }finally{
    setBusy(false);
    if(accepted&&state.current){try{state.current=await api('/api/conversations/'+state.current.id);renderMessages();}catch{}}
    try{await refreshConversations();}catch{}$('prompt').focus();
  }
}
function resizePrompt(){const field=$('prompt');field.style.height='auto';field.style.height=Math.min(field.scrollHeight,180)+'px';}
$('composer').addEventListener('submit',event=>{event.preventDefault();send();});
$('prompt').addEventListener('input',resizePrompt);
$('prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();send();}});
$('stop-button').addEventListener('click',stop);
$('new-chat').addEventListener('click',newConversation);
$('chat-search').addEventListener('input',renderConversations);
$('menu-button').addEventListener('click',()=>$('sidebar').classList.toggle('open'));
document.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='k'){event.preventDefault();newConversation();}});
for(const button of document.querySelectorAll('[data-prompt]'))button.addEventListener('click',()=>{$('prompt').value=button.dataset.prompt;resizePrompt();$('prompt').focus();});
for(const name of ['documents','web','code'])$(name+'-toggle').addEventListener('click',()=>{if(state.busy)return;if(name!=='documents'&&prefs.provider!=='openai'){toast('Choose OpenAI in Settings to use hosted tools.');openSettings();return;}prefs[name]=!prefs[name];persist('atlas.preferences',prefs);updatePrefsUI();});
function inspector(tab){$('inspector').hidden=false;document.querySelector('.app').classList.add('inspector-open');$('documents-panel').hidden=tab!=='documents';$('memories-panel').hidden=tab!=='memories';$('documents-tab').classList.toggle('active',tab==='documents');$('memories-tab').classList.toggle('active',tab==='memories');$('sidebar').classList.remove('open');}
$('knowledge-button').addEventListener('click',()=>inspector('documents'));
$('memory-button').addEventListener('click',()=>inspector('memories'));
$('document-suggestion').addEventListener('click',()=>inspector('documents'));
$('documents-tab').addEventListener('click',()=>inspector('documents'));
$('memories-tab').addEventListener('click',()=>inspector('memories'));
$('close-inspector').addEventListener('click',()=>{$('inspector').hidden=true;document.querySelector('.app').classList.remove('inspector-open');});
async function refreshContext(){
  [state.documents,state.memories]=await Promise.all([api('/api/documents'),api('/api/memories')]);
  $('document-count').textContent=state.documents.length;$('memory-count').textContent=state.memories.length;
  $('embedding-status').textContent=state.config.embedding_provider==='none'?'Keyword retrieval is active. Enable embeddings in .env for hybrid semantic search.':'Hybrid retrieval · '+state.config.embedding_provider+' embeddings';
  $('document-list').replaceChildren();if(!state.documents.length)$('document-list').append(element('div','empty-panel','Your knowledge base is ready.\nAdd a document to get started.'));
  for(const doc of state.documents){const card=element('div','context-card'),top=element('div','context-card-top'),button=element('button','','×');button.setAttribute('aria-label','Delete '+doc.name);button.addEventListener('click',()=>confirmAction('Delete document?',`Remove ${doc.name} and its search index?`,async()=>{await remove('/api/documents/'+doc.id);await refreshContext();}));top.append(element('strong','',doc.name),button);card.append(top,element('small','',`${doc.chunks} passage${doc.chunks===1?'':'s'} · ${doc.characters.toLocaleString()} characters`));$('document-list').append(card);}
  $('memory-list').replaceChildren();if(!state.memories.length)$('memory-list').append(element('div','empty-panel','No saved memory yet.\nAdd a preference or a useful fact.'));
  for(const memory of state.memories){const card=element('div','context-card'),top=element('div','context-card-top'),button=element('button','','×');button.setAttribute('aria-label','Delete memory');button.addEventListener('click',()=>confirmAction('Forget this memory?','It will no longer be included in future model requests.',async()=>{await remove('/api/memories/'+memory.id);await refreshContext();}));top.append(element('div','memory-content',memory.content),button);card.append(top);$('memory-list').append(card);}
}
function readBase64(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(new Error('Unable to read the file.'));reader.readAsDataURL(file);});}
$('upload-document').addEventListener('click',()=>$('document-file').click());
$('document-file').addEventListener('change',async()=>{const files=[...$('document-file').files];$('document-file').value='';$('upload-document').disabled=true;try{for(const file of files){if(file.size>4*1024*1024)throw new Error(file.name+' exceeds 4 MiB.');$('upload-document').querySelector('strong').textContent='Indexing '+file.name+'…';await post('/api/documents',{name:file.name,data:await readBase64(file)});await refreshContext();}toast('Documents added to your knowledge base.');}catch(error){toast(error.message);}finally{$('upload-document').disabled=false;$('upload-document').querySelector('strong').textContent='Add documents';}});
$('memory-form').addEventListener('submit',async event=>{event.preventDefault();const text=$('memory-input').value.trim();if(!text)return;try{await post('/api/memories',{content:text});$('memory-input').value='';await refreshContext();toast('Memory saved.');}catch(error){toast(error.message);}});
function renderAttachments(){const tray=$('attachment-tray');tray.replaceChildren();for(const item of state.attachments){const chip=element('div','attachment-chip'),button=element('button','','×');button.setAttribute('aria-label','Remove '+item.name);button.addEventListener('click',()=>{state.attachments=state.attachments.filter(value=>value.id!==item.id);renderAttachments();});chip.append(element('span','','▧ '+item.name),button);tray.append(chip);}}
$('attach-button').addEventListener('click',()=>$('image-file').click());
$('image-file').addEventListener('change',async()=>{const files=[...$('image-file').files];$('image-file').value='';try{if(state.attachments.length+files.length>4)throw new Error('Attach at most four images.');for(const file of files){if(file.size>4*1024*1024)throw new Error('Images are limited to 4 MiB.');const image=await post('/api/images',{name:file.name,data:await readBase64(file)});state.attachments.push(image);renderAttachments();}toast('Image attached. The selected model must support vision.');}catch(error){toast(error.message);}});
function providerHelp(){
  const provider=$('provider').value;
  $('provider-help').textContent=provider==='openai'?(state.config.openai_configured?'API key configured on the server. Model and tool access depend on your account.':'Set OPENAI_API_KEY in .env and restart the server. API requests are billed by the provider.'):
    provider==='demo'?'Interface demonstration only. Responses are fixed text, not AI-generated.':'Start Ollama and install a model first. For the default: ollama pull qwen3:8b. Images require a vision-capable model.';
  $('reasoning').disabled=provider!=='openai';$('temperature').disabled=provider==='openai';
}
function openSettings(){
  for(const [id,key] of [['provider','provider'],['model','model'],['max-tokens','max_tokens'],['reasoning','reasoning'],['temperature','temperature'],['custom-system','system']])$(id).value=prefs[key];
  $('temperature-label').textContent=prefs.temperature;$('test-result').textContent='';providerHelp();$('settings-dialog').showModal();
}
$('settings-button').addEventListener('click',openSettings);$('model-button').addEventListener('click',openSettings);
$('provider').addEventListener('change',()=>{$('model').value=$('provider').value==='openai'?state.config.openai_model:$('provider').value==='demo'?'interface-demo':state.config.ollama_model;$('test-result').textContent='';providerHelp();});
$('temperature').addEventListener('input',()=>$('temperature-label').textContent=$('temperature').value);
$('test-connection').addEventListener('click',async()=>{$('test-result').textContent='Checking connection…';$('test-connection').disabled=true;try{const result=await post('/api/connection',{provider:$('provider').value,model:$('model').value});$('test-result').textContent=(result.ok?'✓ ':'')+result.message+(result.models.length?'\nModels: '+result.models.slice(0,12).join(', '):'');}catch(error){$('test-result').textContent=error.message;}finally{$('test-connection').disabled=false;}});
$('settings-form').addEventListener('submit',event=>{event.preventDefault();if(state.busy){toast('Finish the current reply before changing models.');return;}prefs={...prefs,provider:$('provider').value,model:$('model').value.trim(),max_tokens:Number($('max-tokens').value),reasoning:$('reasoning').value,temperature:Number($('temperature').value),system:$('custom-system').value};if(prefs.provider!=='openai'){prefs.web=false;prefs.code=false;}persist('atlas.preferences',prefs);updatePrefsUI();$('settings-dialog').close();toast('Preferences saved.');});
for(const button of document.querySelectorAll('[data-close]'))button.addEventListener('click',()=>$(button.dataset.close).close());
let actionCallback=null;
function resetAction(){const footer=$('action-form').querySelector('.dialog-footer');footer.querySelector('.delete-conversation')?.remove();$('action-submit').classList.remove('danger');}
function confirmAction(title,description,callback){resetAction();$('action-title').textContent=title;$('action-label').hidden=true;$('action-input').required=false;$('action-description').textContent=description;$('action-submit').textContent='Delete';$('action-submit').classList.add('danger');actionCallback=callback;$('action-dialog').showModal();}
function manageConversation(item){if(state.busy){toast('Finish the current reply first.');return;}resetAction();$('action-title').textContent='Conversation';$('action-label').hidden=false;$('action-input').required=true;$('action-input').value=item.title;$('action-description').textContent='Rename this conversation or remove it from your workspace.';$('action-submit').textContent='Save title';actionCallback=async()=>{await post('/api/conversations/'+item.id,{title:$('action-input').value.trim()});if(state.current?.id===item.id)state.current.title=$('action-input').value.trim();await refreshConversations();renderMessages();};const deletion=element('button','secondary-button delete-conversation','Delete conversation');deletion.type='button';deletion.addEventListener('click',()=>{$('action-dialog').close();confirmAction('Delete conversation?','This removes its messages and unreferenced image attachments.',async()=>{await remove('/api/conversations/'+item.id);if(state.current?.id===item.id)state.current=null;await refreshConversations();renderMessages();});});$('action-form').querySelector('.dialog-footer').prepend(deletion);$('action-dialog').showModal();}
$('action-form').addEventListener('submit',async event=>{event.preventDefault();try{await actionCallback?.();$('action-dialog').close();}catch(error){toast(error.message);}});
$('export-button').addEventListener('click',()=>{if(!state.current?.messages.length){toast('Start a conversation before exporting.');return;}let text='# '+state.current.title+'\n\n';for(const message of state.current.messages){text+='## '+(message.role==='user'?'You':'Atlas')+'\n\n'+message.content+'\n\n';for(const source of message.metadata?.sources||[])text+=(source.kind==='web'?`Source: ${source.title} — ${source.url}`:`[${source.label}] ${source.title}, passage ${source.chunk}`)+'\n';text+='\n';}const url=URL.createObjectURL(new Blob([text],{type:'text/markdown'})),link=document.createElement('a');link.href=url;link.download=state.current.title.replace(/[^a-z0-9_-]/gi,'_').slice(0,60)+'.md';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('theme-button').addEventListener('click',()=>{document.body.classList.toggle('dark');persist('atlas.dark',document.body.classList.contains('dark'));});
async function init(){
  document.body.classList.toggle('dark',saved('atlas.dark',false));
  try{state.config=await api('/api/config');if(state.config.demo_enabled){const option=element('option','','Interface demo · no AI');option.value='demo';$('provider').append(option);}
    if(!prefs.model){prefs.provider=state.config.demo_enabled?'demo':state.config.openai_configured?'openai':'ollama';prefs.model=prefs.provider==='demo'?'interface-demo':prefs.provider==='openai'?state.config.openai_model:state.config.ollama_model;}
    if(prefs.provider==='demo'&&!state.config.demo_enabled){prefs.provider='ollama';prefs.model=state.config.ollama_model;}
    if(prefs.provider!=='openai'){prefs.web=false;prefs.code=false;}
    updatePrefsUI();await Promise.all([refreshConversations(),refreshContext()]);renderMessages();resizePrompt();
  }catch(error){toast('Unable to load Atlas: '+error.message);}
}
init();
