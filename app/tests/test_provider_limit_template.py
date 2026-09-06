"""Execute quota sample merging and polling races using the template's JS."""
from pathlib import Path
import shutil
import subprocess
import unittest

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "dashboard.html"


class ProviderLimitTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = TEMPLATE.read_text()

    def node(self, source):
        if not shutil.which("node"):
            self.skipTest("node unavailable")
        result = subprocess.run(["node", "-"], input=source, text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def helper(self):
        return "function mergeLimitBudget(" + self.template.split(
            "function mergeLimitBudget(", 1)[1].split("function initRateLimits(){", 1)[0]

    def test_snapshot_fetch_precedes_chart_and_payload_and_limit_init_precedes_render(self):
        self.assertLess(self.template.index("window.tfLimitSnapshot ="),
                        self.template.index('defer src="/static/chart'))
        boot = self.template.split("function boot(){", 1)[1].split("\n}", 1)[0]
        self.assertLess(boot.index("initRateLimits();"), boot.index("setMode("))
        self.assertIn("controller.abort();", self.template)

    def test_provider_uses_shared_anthropic_budget_stats_and_explicit_caveat(self):
        shared = self.template.split("function providerWindowStats", 1)[1].split(
            "function buildReportedProviderGroups", 1)[0]
        self.assertIn("budgetStats(win.window_cost, pct)", shared)
        self.assertIn("API-equivalent", shared)
        block = self.template.split("function buildReportedProviderGroups", 1)[1].split(
            "return html;", 1)[0]
        self.assertIn("stats: providerStats", block)
        self.assertIn("not a dollar allowance", block)
        self.assertIn("windowState(win.resets_at, observation, nowMs)", block)

    def test_provider_dollar_render_without_oauth_uses_outer_shared_helper(self):
        # Extract only the shared render scope, stopping BEFORE the OAuth IIFE.
        # A helper nested in that IIFE is not callable for either scope.
        shared = self.template.split("  function providerGroup(", 1)[1].split(
            "  /* ── Personal OAuth limit gauges", 1)[0]
        self.node("const assert = require('assert');\n" + self.helper() + """
const esc = String, fC = n => '$' + n.toFixed(2);
const windowPace = () => ({expected:10,projected:20,ticks:[]});
let gauges=[];
function buildGauge(label,pct,reset,expected,options) { gauges.push(options); return label; }
""" + "function providerGroup(" + shared + """
for(const scope of ['enterprise','personal']) {
  const wb = {providers:{codex:{updated_at_epoch:Date.now()/1000 - 60,
    windows:[{key:'primary',label:'7-day limit',pct:10,window_seconds:604800,
      resets_at:new Date(Date.now()+86400000).toISOString(),window_cost:10}]}}};
  assert.equal(wb.oauth,undefined);
  // A full response has been accepted into observation-aligned detail memory.
  const render = () => buildReportedProviderGroups(wb.providers, {memory:{providers:{
    codex:{at:wb.providers.codex.updated_at_epoch,sample:wb.providers.codex,cells:{}}
  }}});
  let html = render();
  assert(html.includes('Codex')); assert(html.includes('7-day limit'));
  assert.equal(gauges.at(-1).stats.length,3);
  assert.equal(gauges.at(-1).stats[1].value,'~$100.00');
  assert.equal(gauges.at(-1).stats[2].value,'~$90.00');
  assert(gauges.at(-1).note.includes('not a dollar allowance'));
  wb.providers.codex.windows[0].resets_at = new Date(Date.now()-1000).toISOString();
  render();
  assert.equal(gauges.at(-1).stats.length,0);
  assert.equal(gauges.at(-1).state,'expired');
  wb.providers.codex.windows[0].resets_at = new Date(Date.now()+86400000).toISOString();
  wb.providers.codex.updated_at_epoch = Date.now()/1000 - 3601;
  render();
  assert.equal(gauges.at(-1).state,'stale');
  assert.equal(gauges.at(-1).stats.length,1,'stale: measured spend only, no capacity inference');
  wb.providers.codex.updated_at_epoch = Date.now()/1000 + 120;
  render();
  assert.equal(gauges.at(-1).state,'unverified');
  assert.equal(gauges.at(-1).stats.length,1,'future observation never feeds inference');
}
""")

    def test_snapshot_merge_preserves_same_sample_dollars_but_not_new_sample(self):
        self.node("const assert = require('assert');\n" + self.helper() + """
const win = {key:'primary', pct:10, resets_at:'future', window_seconds:604800};
const old = {week_cost:12, monthly_budget:{budget_usd:100},
  oauth:{updated_at_epoch:5, weekly_pct:10, limit_window:{cost:12},
    five_hour_window:{cost:3}, buckets:[{key:'scoped:opus',pct:10,resets_at:'expired',window_cost:4}]},
  providers:{codex:{updated_at_epoch:5, windows:[{...win,window_cost:10}],month_cost:20},
             'opencode-zen':{windows:[],month_cost:2}}};
const cheap = {oauth:{updated_at_epoch:5,weekly_pct:10,
    buckets:[{key:'scoped:opus',pct:10,resets_at:'expired'}]},
  providers:{codex:{updated_at_epoch:5,windows:[win]}}};
let merged = mergeLimitBudget(old,cheap,true);
assert.equal(merged.providers.codex.windows[0].window_cost,10);
assert.equal(merged.oauth.limit_window.cost,12);
assert.equal(merged.providers['opencode-zen'].month_cost,2);
assert.equal(merged.monthly_budget.budget_usd,100);
const cleared = mergeLimitBudget(old,{providers:{}},false);
assert.equal(cleared.monthly_budget,undefined); // full response confirms budget removal
const expired = mergeLimitBudget(old,cheap,false);
assert.equal(expired.oauth.limit_window,undefined); // omission removes expired derived fields
assert.equal(expired.oauth.five_hour_window,undefined);
assert.equal(expired.oauth.buckets[0].window_cost,undefined);
assert.equal(expired.providers.codex.windows[0].window_cost,undefined);
cheap.providers.codex.updated_at_epoch = 6;
cheap.oauth.updated_at_epoch = 6;
merged = mergeLimitBudget(old,cheap,true);
assert.equal(merged.providers.codex.windows[0].window_cost,undefined);
assert.equal(merged.oauth.limit_window,undefined);
merged = mergeLimitBudget(merged,old,false);
assert.equal(merged.providers.codex.updated_at_epoch,6);
assert.equal(merged.providers.codex.windows[0].window_cost,undefined);
assert.equal(mergeLimitBudget(old,{providers:{}},true).providers.codex.windows.length,0);
""")

    def test_snapshot_merge_keeps_independent_month_totals_across_new_samples(self):
        self.node("const assert = require('assert');\n" + self.helper() + """
const win = {key:'primary', pct:10, resets_at:'future', window_seconds:604800};
const old = {providers:{codex:{updated_at_epoch:5, windows:[{...win,window_cost:10}],month_cost:20}}};
const newer = {providers:{codex:{updated_at_epoch:6, windows:[{...win,pct:11}]}}};
const merged = mergeLimitBudget(old,newer,true);
assert.equal(merged.providers.codex.windows[0].window_cost,undefined); // new sample, old dollars refused
assert.equal(merged.providers.codex.month_cost,20); // month total is not part of the sample
assert.equal(merged.providers.codex.windows[0].pct,11);
assert.equal(mergeLimitBudget(old,{providers:{codex:{updated_at_epoch:6,windows:[win]}}},false).providers.codex.month_cost,undefined);
""")

    def test_poll_failure_timeout_and_old_generation_cannot_overwrite_gauges(self):
        poll = self.template.split("    var rlUrl = '/api/rate-limits?scope='", 1)[1]
        poll = "var rlUrl = '/api/rate-limits?scope='" + poll.split("\n}\n\n/* ---- Single global", 1)[0]
        self.node("const assert = require('assert');\n" + self.helper() + """
const document = {hidden:false}, window = {tfLimitSnapshot:null}, TF_SCOPE='personal';
let rendered=[], errors=0, throwNext=false, requested=[];
const feeds={};
const tfRefresh={add:(name,config)=>feeds[name]=config,request:name=>requested.push(name)};
const TokenfoldRefresh={fetchJSON:()=>Promise.resolve(null)};
function renderAllRateLimits(data) {
  if(throwNext) { throwNext=false; throw new Error('render regression'); }
  rendered.push(data);
}
function showLimitsError() { errors++; }
const sample = (at,cost) => ({weekly_budget:{providers:{codex:{updated_at_epoch:at,
  windows:[{key:'primary',pct:10,resets_at:'future',window_seconds:604800,
  ...(cost ? {window_cost:cost}: {})}]}}}});
const flush = () => new Promise(resolve=>setImmediate(resolve));
""" + poll + """
(async()=>{
  pollLimits(); assert.deepEqual(requested,['limitSnapshots','limits']);
  feeds.limitSnapshots.apply(sample(1)); assert.equal(rendered.length,1);
  feeds.limits.apply(sample(3,30));
  feeds.limitSnapshots.apply(sample(3)); // late cheap response retains enrichment
  feeds.limitSnapshots.apply(sample(2)); feeds.limits.apply(sample(2,20));
  assert.equal(rendered.at(-1).providers.codex.updated_at_epoch,3);
  assert.equal(rendered.at(-1).providers.codex.windows[0].window_cost,30);
  throwNext=true;
  assert.throws(()=>feeds.limitSnapshots.apply(sample(4)),/render regression/);
  assert.equal(errors,1);
  assert.equal(limitsState.providers.codex.updated_at_epoch,3);
  feeds.limits.apply(sample(4,40));
  assert.equal(limitsState.providers.codex.windows[0].window_cost,40);
  assert.equal(feeds.limits.interval,60000);
  assert.equal(feeds.limitSnapshots.interval,15000);
})().catch(e=>{console.error(e);process.exitCode=1;});
""")

    def test_failed_head_start_snapshot_falls_back_and_new_sample_requests_detail_now(self):
        poll = self.template.split("    var rlUrl = '/api/rate-limits?scope='", 1)[1]
        poll = "var rlUrl = '/api/rate-limits?scope='" + poll.split("\n}\n\n/* ---- Single global", 1)[0]
        self.node("const assert = require('assert');\n" + self.helper() + """
const document = {hidden:false}, window = {tfLimitSnapshot:Promise.resolve(null)}, TF_SCOPE='personal';
let fetched=[], requested=[];
const feeds={};
const tfRefresh={add:(name,config)=>feeds[name]=config,request:(name,now)=>requested.push([name,!!now])};
const TokenfoldRefresh={fetchJSON:(url)=>{fetched.push(url);return Promise.resolve({weekly_budget:{providers:{}}});}};
function renderAllRateLimits() {}
function showLimitsError() { throw new Error('error flash on fallback'); }
const sample = at => ({weekly_budget:{providers:{codex:{updated_at_epoch:at,
  windows:[{key:'primary',pct:10,resets_at:'future',window_seconds:604800}]}}}});
""" + poll + """
(async()=>{
  const data = await feeds.limitSnapshots.load({});
  assert.deepEqual(fetched,[snapshotUrl],'null head-start must fall back to a real fetch');
  assert.ok(data && data.weekly_budget);
  window.tfLimitSnapshot = null; fetched=[];
  await feeds.limitSnapshots.load({});
  assert.deepEqual(fetched,[snapshotUrl]);
  feeds.limitSnapshots.apply(sample(1));
  feeds.limitSnapshots.apply(sample(2));
  assert.deepEqual(requested.at(-1),['limits',true],'a new sample fetches its detail without the minimum interval');
})().catch(e=>{console.error(e);process.exitCode=1;});
""")
