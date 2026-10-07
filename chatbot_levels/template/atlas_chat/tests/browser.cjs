// Optional browser smoke test: npm install --no-save playwright
// Run against an isolated server started with --demo.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({headless:true, args:['--no-sandbox']});
  const context = await browser.newContext({viewport:{width:1440,height:960},acceptDownloads:true});
  const page = await context.newPage();
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(()=>{window.cspViolations=[];document.addEventListener('securitypolicyviolation',event=>window.cspViolations.push(event.violatedDirective));});
  const base=process.env.ATLAS_TEST_URL||'http://127.0.0.1:8765';
  try {
    await page.goto(base);
    await page.locator('#model-label').filter({hasText:'interface-demo'}).waitFor();
    await page.screenshot({path:path.join(__dirname,'..','preview.png'),fullPage:true});
    await page.locator('#knowledge-button').click();
    await page.locator('#document-file').setInputFiles({name:'browser-notes.txt',mimeType:'text/plain',buffer:Buffer.from('Jupiter is the largest planet in the solar system. Its Great Red Spot is a storm.')});
    await page.locator('#document-list .context-card').filter({hasText:'browser-notes.txt'}).waitFor();
    await page.locator('#memories-tab').click();
    await page.locator('#memory-input').fill('I prefer concise explanations with examples.');
    await page.locator('#memory-form button').click();
    await page.locator('#memory-list .context-card').filter({hasText:'concise explanations'}).waitFor();
    await page.locator('#close-inspector').click();
    await page.locator('#prompt').fill('Tell me about Jupiter.');
    await page.locator('#send-button').click();
    await page.locator('.message.assistant .message-actions').waitFor();
    assert.match(await page.locator('.message.assistant .message-content').innerText(),/Interface demo/);
    await page.locator('.message.assistant .source-chip').filter({hasText:'browser-notes'}).click();
    assert.match(await page.locator('#source-text').innerText(),/largest planet/);
    await page.locator('[data-close="source-dialog"]').click();
    const count=await page.locator('.message').count();
    await page.reload();
    await page.locator('#conversations .conversation-select').first().click();
    await page.locator('.message.assistant .message-actions').waitFor();
    assert.equal(await page.locator('.message').count(),count);
    await page.locator('#theme-button').click();
    assert.equal(await page.locator('body').evaluate(node=>node.classList.contains('dark')),true);
    await page.locator('#theme-button').click();
    const downloadPromise=page.waitForEvent('download');
    await page.locator('#export-button').click();
    const download=await downloadPromise;
    assert.match(download.suggestedFilename(),/\.md$/);
    // Generated text must never become active HTML or executable links.
    const rendered=await page.evaluate(()=>{
      const host=document.createElement('div');host.append(markdown('<img src=x onerror="window.compromised=true">\n\n[bad](javascript:alert(1))\n\n```html\n<script>alert(1)</script>\n```'));
      return {images:host.querySelectorAll('img').length,scripts:host.querySelectorAll('script').length,bad:host.querySelectorAll('a[href^="javascript:"]').length,text:host.textContent};
    });
    assert.equal(rendered.images,0);assert.equal(rendered.scripts,0);assert.equal(rendered.bad,0);assert.match(rendered.text,/<img/);
    await page.locator('#model-button').click();
    await page.locator('#test-connection').click();
    await page.locator('#test-result').filter({hasText:'Demo mode'}).waitFor();
    await page.locator('[data-close="settings-dialog"]').first().click();
    await page.setViewportSize({width:390,height:844});
    await page.locator('#menu-button').click();
    await page.locator('#new-chat').click();
    await page.locator('#welcome').waitFor({state:'visible'});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.screenshot({path:path.join(__dirname,'..','mobile-preview.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    assert.deepEqual(await page.evaluate(()=>window.cspViolations),[]);
    console.log('Browser smoke test passed: chat streaming, retrieval passages, memory, reload, export, theme, model settings, escaping, and mobile layout.');
  } finally {
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
