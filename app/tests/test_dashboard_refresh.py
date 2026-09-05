"""Executable lifecycle regressions; no browser or third-party JS dependencies."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DashboardRefreshTest(unittest.TestCase):
    def test_resource_lifecycle(self):
        result = subprocess.run(['node', str(ROOT / 'app/tests/dashboard_refresh.test.cjs')],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_same_revision_render_failure_retries_until_success(self):
        template = (ROOT / 'templates/dashboard.html').read_text()
        state = 'var knownVersion' + template.split('var knownVersion', 1)[1].split('var tfRefresh', 1)[0]
        resource = "tfRefresh.add('stats', {" + template.split("tfRefresh.add('stats', {", 1)[1].split("tfRefresh.add('version'", 1)[0]
        script = """
const assert=require('assert');
const D={version:1,generation_time:'old'};
let calls=0, feed;
const TF_SCOPE='personal', TokenfoldRefresh={};
const tfRefresh={add:(name,config)=>feed=config,request:()=>{}};
function applyDashboardData(data){Object.assign(D,data);if(++calls===1)throw Error('render');}
""" + state + resource + """
const next={version:1,generation_time:'new'};
assert.throws(()=>feed.apply(next),/render/);
feed.apply(next);
assert.equal(calls,2,'same revision must not skip a failed partial repaint');
"""
        result = subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_authoritative_render_replaces_optional_fields_without_mutating_arrays(self):
        template = (ROOT / 'templates/dashboard.html').read_text()
        render = 'function applyDashboardData' + template.split('function applyDashboardData', 1)[1].split('/* ---- Responsive:', 1)[0]
        script = """
const assert=require('assert');
const D={ghost:{old:true},model_breakdown:[{model:'old'}]};
let mb=D.model_breakdown, outPriceMap={}, MODEL_ORDER=[];
const document={getElementById:()=>null};
const lsGet=()=>null;
function renderMode(){assert.equal(mb.length,1);assert.equal(mb[0].model,'new');}
function renderHeatmap(){} function renderHourly(){} function renderSessions(){}
function renderBillingMeter(){} function renderPricing(){} function renderBenchmarks(){}
function refreshEnvModal(){} function updateStatus(){}
""" + render + """
const incoming={model_breakdown:[{model:'new'}],daily:[],today:{},cards:{},version:2};
applyDashboardData(incoming);
assert.equal(D.ghost,undefined);
assert.equal(incoming.model_breakdown.length,1);
assert.equal(D.model_breakdown,mb);
assert.throws(()=>applyDashboardData({version:3}),/bad stats payload/);
"""
        result = subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
