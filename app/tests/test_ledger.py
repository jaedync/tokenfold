"""GET /api/ledger and GET /api/sessions/{id} through real ingest."""
import time
from datetime import datetime, timezone
from unittest.mock import patch

from app.tests._support import TempDBTestCase

PERSONAL = {'account_email': 'me@example.test', 'plan': 'pro'}
WORK = {'account_email': 'me@corp.test', 'plan': 'enterprise'}


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace('+00:00', 'Z')


class LedgerTestBase(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        rebuild = patch('app.aggregator.trigger_eager_rebuild')
        rebuild.start()
        self.addCleanup(rebuild.stop)
        lock = patch('app.config.LOCKED_SCOPE', None)
        lock.start()
        self.addCleanup(lock.stop)
        import app.ledger as ledger
        import app.ledger_days as ledger_days
        ledger._cache._entries.clear()
        ledger_days._days._entries.clear()
        self.t0 = float(int(time.time()) - 7200)
        self.http = self.client()
        self.http.__enter__()
        self.addCleanup(self.http.__exit__, None, None, None)

    # ── ingest helpers ──
    def cc(self, session_file, records, account=PERSONAL, machine='Laptop.local',
           project='-home-me-dev-dashboard'):
        body = {'machine': machine, 'project_dir': project,
                'session_file': session_file, 'events': records, **account}
        r = self.http.post('/api/ingest', json=body, headers={'X-API-Key': self.api_key})
        self.assertEqual(r.status_code, 200, r.text)

    def pi(self, session_file, events, machine='devbox', project='/home/me/dev/api-server',
           account_class='personal'):
        body = {'machine': machine, 'account_class': account_class,
                'project_dir': project, 'session_file': session_file, 'events': events}
        r = self.http.post('/api/ingest/pi', json=body, headers={'X-API-Key': self.api_key})
        self.assertEqual(r.status_code, 200, r.text)

    def cc_user(self, uuid, sid, at, agent=None, human=True):
        rec = {'uuid': uuid, 'sessionId': sid, 'type': 'user', 'timestamp': iso(at),
               'userType': 'external',
               'message': {'role': 'user', 'content': 'go' if human else '<cmd>'}}
        if agent:
            rec.update(agentId=agent, isSidechain=True)
        return rec

    def cc_reply(self, uuid, sid, at, req, model='claude-opus-4-8', inp=1_000_000,
                 tools=(), agent=None, msg_id=None):
        content = [{'type': 'tool_use', 'id': tid, 'name': name, 'input': {}}
                   for tid, name in tools]
        rec = {'uuid': uuid, 'sessionId': sid, 'type': 'assistant', 'timestamp': iso(at),
               'requestId': req,
               'message': {'id': msg_id or 'id_' + req, 'model': model,
                           'usage': {'input_tokens': inp}, 'content': content}}
        if agent:
            rec.update(agentId=agent, isSidechain=True)
        return rec

    def pi_event(self, eid, sid, at, kind='assistant', cost=1.0, model='claude-opus-4-8',
                 provider='anthropic', request_id=None, agent=None, tools=()):
        e = {'event_id': eid, 'session_id': sid, 'timestamp': iso(at), 'kind': kind}
        if kind == 'assistant':
            e.update(provider=provider, model=model, request_id=request_id or 'req-' + eid,
                     usage={'input': 1000, 'output': 10, 'cost_total': cost})
            if tools:
                e['tools'] = [{'tool_use_id': eid + n, 'name': n} for n in tools]
        if agent:
            e.update(agent_id=agent, is_sidechain=True)
        return e

    def ledger(self, scope='personal', **params):
        q = {'scope': scope, 'from': self.t0 - 60, 'to': self.t0 + 3600}
        q.update(params)
        r = self.http.get('/api/ledger', params=q)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def session(self, sid, scope='personal'):
        return self.http.get('/api/sessions/' + sid, params={'scope': scope})


class ClaudeCodeLedgerTest(LedgerTestBase):
    def setUp(self):
        super().setUp()
        t = self.t0
        self.cc('cc-1.jsonl', [
            self.cc_user('u1', 'cc-1', t),
            self.cc_reply('a1', 'cc-1', t + 10, 'r1', tools=[('tu1', 'Task')]),
            self.cc_user('su1', 'cc-1', t + 20, agent='agent-a', human=False),
            self.cc_reply('sa1', 'cc-1', t + 30, 'r2', model='claude-sonnet-4-6',
                          tools=[('tu2', 'Bash'), ('tu3', 'Edit')], agent='agent-a'),
            self.cc_user('u2', 'cc-1', t + 100),
            self.cc_reply('a2', 'cc-1', t + 110, 'r3', inp=200_000,
                          tools=[('tu4', 'Write')]),
        ])
        self.conn.execute("INSERT INTO session_titles(session_id, title) VALUES('cc-1', 'Add drawer')")
        self.conn.commit()
        self.cc('ent.jsonl', [self.cc_reply('e1', 'ent', t + 5, 'r9')], account=WORK)

    def test_session_row_splits_orchestrator_and_agent_cost(self):
        body = self.ledger()
        self.assertEqual([s['id'] for s in body['sessions']], ['cc-1'])
        s = body['sessions'][0]
        self.assertEqual(s['title'], 'Add drawer')
        self.assertEqual(s['client'], 'claude-code')
        self.assertEqual(s['machine'], 'laptop')
        self.assertEqual(s['project'], 'dev-dashboard')
        self.assertAlmostEqual(s['orch_cost_usd'], 6.0)   # 1.2M Opus 4.8 input
        self.assertAlmostEqual(s['agent_cost_usd'], 3.0)  # 1M Sonnet 4.6 input
        self.assertAlmostEqual(s['cost_usd'], 9.0)
        self.assertEqual(s['model'], 'Opus 4.8')
        self.assertEqual(s['prompts'], 2)
        self.assertEqual(s['start'], self.t0)
        self.assertEqual(s['end'], self.t0 + 110)
        self.assertEqual(s['active_s'], 110)  # gaps 10 + 90 + 10, all under 300 s
        self.assertFalse(s['live'])
        self.assertEqual(len(s['runs']), 1)
        run = s['runs'][0]
        self.assertEqual(run['id'], 'agent-a')
        self.assertIsNone(run['parent'])
        self.assertEqual(run['model'], 'Sonnet 4.6')
        self.assertAlmostEqual(run['cost_usd'], 3.0)
        self.assertEqual(run['tools'], 2)
        self.assertIsNone(run['label'])  # Task descriptions do not survive ingest
        self.assertEqual((run['start'], run['end']), (self.t0 + 20, self.t0 + 30))

    def test_totals_and_cost_match_window_cost(self):
        from app.config import scope_predicate  # noqa: F401 - scope import path check
        from app.cost_windows import compute_window_cost
        from app.db import get_conn
        body = self.ledger()
        expected = compute_window_cost(get_conn(), self.t0 - 60, self.t0 + 3600, 'personal')
        self.assertAlmostEqual(body['totals']['cost_usd'], expected)
        self.assertEqual(body['totals']['sessions'], 1)
        self.assertEqual(body['totals']['runs'], 1)
        self.assertEqual(body['totals']['prompts'], 2)
        self.assertEqual(body['totals']['agent_active_s'], 10)

    def test_scope_partition(self):
        body = self.ledger(scope='enterprise')
        self.assertEqual([s['id'] for s in body['sessions']], ['ent'])
        self.assertEqual(self.session('ent').status_code, 404)
        self.assertEqual(self.session('cc-1', scope='enterprise').status_code, 404)

    def test_mixed_scope_session_counts_only_in_scope_rows(self):
        # An account switch mid-session: the work half stays out of personal.
        t = self.t0 + 1000
        self.cc('cc-1b.jsonl', [self.cc_user('w-u', 'cc-1', t),
                                self.cc_reply('w-a', 'cc-1', t + 5, 'rw')], account=WORK)
        s = self.ledger()['sessions'][0]
        self.assertAlmostEqual(s['cost_usd'], 9.0)
        self.assertEqual(s['prompts'], 2)
        self.assertEqual(s['end'], self.t0 + 110)
        w = {x['id']: x for x in self.ledger(scope='enterprise')['sessions']}['cc-1']
        self.assertAlmostEqual(w['cost_usd'], 5.0)
        self.assertEqual((w['prompts'], w['active_s']), (1, 5))

    def test_daily_by_model_from_rollup(self):
        body = self.ledger()
        day = datetime.fromtimestamp(self.t0).astimezone().strftime('%Y-%m-%d')
        days = {d['day']: d['by_model'] for d in body['daily']}
        self.assertIn(day, days)
        self.assertAlmostEqual(sum(days[day].values()), 9.0)
        self.assertAlmostEqual(days[day]['Opus 4.8'], 6.0)

    def test_window_bounds_cost_and_runs(self):
        body = self.ledger(**{'from': self.t0 + 50})
        s = body['sessions'][0]
        self.assertAlmostEqual(s['cost_usd'], 1.0)
        self.assertEqual(s['runs'], [])

    def test_invalid_ranges_are_rejected(self):
        for params in ({'from': 10, 'to': 5}, {'from': 0, 'to': 400 * 86400},
                       {'from': 'nan'}):
            q = {'scope': 'personal', 'from': self.t0, 'to': self.t0 + 10}
            q.update(params)
            self.assertEqual(self.http.get('/api/ledger', params=q).status_code, 400, params)

    def test_failing_part_degrades_to_null(self):
        with patch('app.ledger.daily_by_model', side_effect=RuntimeError('boom')):
            body = self.ledger()
        self.assertIsNone(body['daily'])
        self.assertEqual(len(body['sessions']), 1)

    def test_session_detail(self):
        r = self.session('cc-1')
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d['id'], 'cc-1')
        self.assertEqual(d['prompt_times'], [self.t0, self.t0 + 100])
        models = {m['model']: m for m in d['models']}
        self.assertAlmostEqual(models['Opus 4.8']['cost_usd'], 6.0)
        self.assertAlmostEqual(models['Opus 4.8']['share'], 6 / 9)
        tools = {t['name']: t for t in d['tools']}
        self.assertEqual(tools['Task']['calls'], 1)
        self.assertIsNone(tools['Task']['errors'])  # ingest never records tool errors
        self.assertEqual(d['tokens'], {'input': 2_200_000, 'output': 0,
                                       'cache_read': 0, 'cache_write': 0})
        self.assertEqual(d['file_edits'], 2)  # Edit + Write
        self.assertEqual(d['served_model'], {'state': 'unknown', 'blocks': 0})
        self.assertEqual(len(d['runs']), 1)

    def test_works_without_activity_index(self):
        self.conn.execute("DROP INDEX idx_events_session_activity")
        self.conn.commit()
        s = self.ledger()['sessions'][0]
        self.assertEqual((s['active_s'], s['prompts']), (110, 2))

    def test_session_detail_unknown_is_404(self):
        self.assertEqual(self.session('nope').status_code, 404)


class TwinAndPiLineageTest(LedgerTestBase):
    def setUp(self):
        super().setUp()
        t = self.t0
        # Meridian: the SDK copy shares msg_twin with the Pi original.
        self.cc('twin.jsonl', [
            self.cc_user('tw-u', 'twin', t),
            self.cc_reply('tw-a', 'twin', t + 1, 'r-twin', msg_id='msg_twin')])
        self.pi('bucket/2026-10-14T00-00-00-000Z_P1.jsonl', [
            self.pi_event('p-u', 'P1', t, kind='user'),
            self.pi_event('p-a', 'P1', t + 2, cost=2.0, request_id='msg_twin',
                          tools=('subagent',)),
            self.pi_event('p-b', 'P1', t + 900, cost=1.0)])
        self.pi('subagents/P1/2026-10-14T00-10-00-000Z_be-ledger_aaaa1111.jsonl', [
            self.pi_event('c-a', 'C1', t + 60, cost=4.0, model='claude-sonnet-4-6',
                          agent='be-ledger', tools=('bash', 'edit')),
            self.pi_event('c-b', 'C1', t + 120, cost=0.5, model='claude-sonnet-4-6',
                          agent='be-ledger')])
        self.pi('subagents/C1/2026-10-14T00-20-00-000Z_deep-dive_bbbb2222.jsonl', [
            self.pi_event('g-a', 'G1', t + 200, cost=0.25, agent='deep-dive')])
        self.pi('subagents/MISSING/2026-10-14T00-30-00-000Z_lonely_cccc3333.jsonl', [
            self.pi_event('o-a', 'O1', t + 300, cost=0.75, agent='lonely')])

    def by_suffix(self, body):
        return {s['id'].rsplit(':', 1)[-1]: s for s in body['sessions']}

    def test_twin_excluded_and_children_nest_under_parent(self):
        sessions = self.by_suffix(self.ledger())
        self.assertEqual(set(sessions), {'P1', 'O1'})
        p = sessions['P1']
        self.assertEqual(p['client'], 'pi-agent')
        self.assertEqual(p['project'], 'api-server')
        self.assertAlmostEqual(p['orch_cost_usd'], 3.0)
        self.assertAlmostEqual(p['agent_cost_usd'], 4.75)
        self.assertAlmostEqual(p['cost_usd'], 7.75)
        self.assertEqual(p['prompts'], 1)
        runs = {r['label']: r for r in p['runs']}
        self.assertEqual(set(runs), {'be-ledger', 'deep-dive'})
        self.assertIsNone(runs['be-ledger']['parent'])
        self.assertEqual(runs['deep-dive']['parent'], runs['be-ledger']['id'])
        self.assertEqual(runs['be-ledger']['tools'], 2)
        self.assertEqual(runs['be-ledger']['model'], 'Sonnet 4.6')
        self.assertEqual(p['end'], self.t0 + 900)

    def test_unlinked_child_stands_alone_as_agent_cost(self):
        o = self.by_suffix(self.ledger())['O1']
        self.assertEqual(o['title'], 'lonely')
        self.assertEqual(o['orch_cost_usd'], 0)
        self.assertAlmostEqual(o['agent_cost_usd'], 0.75)
        self.assertEqual([r['label'] for r in o['runs']], ['lonely'])

    def test_session_detail_includes_children(self):
        sid = self.by_suffix(self.ledger())['P1']['id']
        d = self.session(sid).json()
        self.assertAlmostEqual(sum(m['cost_usd'] for m in d['models']), 7.75)
        tools = {t['name']: t['calls'] for t in d['tools']}
        self.assertEqual(tools, {'subagent': 1, 'bash': 1, 'edit': 1})
        self.assertEqual(d['file_edits'], 1)
        self.assertEqual(d['prompt_times'], [self.t0])
        self.assertEqual(len(d['runs']), 2)


class PureHelpersTest(TempDBTestCase):
    def test_active_time_accumulator(self):
        from app.ledger_query import accumulate_activity
        rows = [('s', None, 0.0, 1), ('s', None, 100.0, 0), ('s', None, 500.0, 1),
                ('s', 'a', 10.0, 0), ('s', 'a', 20.0, 0)]
        acc = accumulate_activity(rows)
        self.assertEqual(acc[('s', None)], {'active_s': 100.0, 'start': 0.0,
                                             'end': 500.0, 'prompts': 2})
        self.assertEqual(acc[('s', 'a')]['active_s'], 10.0)

    def test_served_state(self):
        from app.session_detail import served_state
        self.assertEqual(served_state([]), {'state': 'unknown', 'blocks': 0})
        self.assertEqual(served_state([('m', 'm', 3)])['state'], 'as_asked')
        self.assertEqual(served_state([('m', 'm', 3), ('m', None, 1)])['state'], 'partly_hidden')
        self.assertEqual(served_state([('m', None, 1), ('m', 'x', 1)])['state'], 'rerouted')

    def test_pi_child_file(self):
        from app.pi_lineage import parse_child_file
        self.assertEqual(parse_child_file('subagents/P1/2026-10-14T00-10-00-000Z_be-ledger_aa11.jsonl'),
                         ('P1', 'be-ledger'))
        self.assertEqual(parse_child_file('subagents/P1/2026_my_name_x_aa11.jsonl'),
                         ('P1', 'my_name_x'))
        self.assertIsNone(parse_child_file('bucket/2026_P1.jsonl'))

    def test_project_names(self):
        from app.ledger_query import project_names
        names = project_names(['/home/me/dev/api', '/srv/other/api', '-home-me-dev-x',
                               '/home/me', None])
        self.assertEqual(names['/home/me/dev/api'], 'dev/api')
        self.assertEqual(names['/srv/other/api'], 'other/api')
        self.assertEqual(names['-home-me-dev-x'], 'dev-x')
        self.assertEqual(names['/home/me'], '~')


class LedgerDaysTest(TempDBTestCase):
    def test_split_range_keeps_closed_whole_days(self):
        from app.aggregator import TZ
        from app.ledger_days import split_range
        day = datetime(2026, 10, 10, tzinfo=TZ)
        start = day.timestamp() + 3600            # 01:00 on Oct 10
        now = day.timestamp() + 3 * 86400 + 600   # 00:10 on Oct 13
        days, rest = split_range(start, now, now)
        self.assertEqual([d for d, _lo, _hi in days], ['2026-10-11', '2026-10-12'])
        self.assertEqual(rest, [(start, day.timestamp() + 86400),
                                (day.timestamp() + 3 * 86400, now)])

    def test_split_range_handles_dst_day(self):
        from app.aggregator import TZ
        from app.ledger_days import split_range
        start = datetime(2026, 11, 1, tzinfo=TZ).timestamp()   # 25-hour day
        end = datetime(2026, 11, 3, tzinfo=TZ).timestamp()
        days, rest = split_range(start, end, end + 86400)
        self.assertEqual([(d, hi - lo) for d, lo, hi in days],
                         [('2026-11-01', 90000), ('2026-11-02', 86400)])
        self.assertEqual(rest, [])

    def test_merge_does_not_mutate_parts(self):
        from collections import Counter
        from app.ledger_days import merge_threads
        part = {('s', None): {'cost': 1.0, 'models': Counter({'m': 1.0}),
                              'tokens': Counter({'input': 5}), 'start': 1, 'end': 2,
                              'machine': 'x', 'project_dir': None, 'client': 'c',
                              'account': None}}
        merged = merge_threads([part, part])
        self.assertEqual(merged[('s', None)]['cost'], 2.0)
        self.assertEqual(merged[('s', None)]['models']['m'], 2.0)
        self.assertEqual(part[('s', None)]['cost'], 1.0)
        self.assertEqual(part[('s', None)]['models']['m'], 1.0)


class LedgerDayCacheTest(LedgerTestBase):
    def test_closed_day_is_reused_until_its_rollup_changes(self):
        import app.ledger_days as days
        days._days._entries.clear()
        t = self.t0 - 3 * 86400
        self.cc('old.jsonl', [self.cc_reply('o1', 'old', t, 'ro1')])
        q = {'from': t - 86400 * 2, 'to': self.t0 + 60}
        first = self.ledger(**q)['totals']['cost_usd']
        import app.ledger as ledger
        ledger._cache._entries.clear()
        with patch('app.ledger_days.billable_groups', wraps=days.billable_groups) as scan:
            self.assertAlmostEqual(self.ledger(**q)['totals']['cost_usd'], first)
            closed_scans = scan.call_count
        self.assertLessEqual(closed_scans, 2)  # only the edge pieces, no closed day
        ledger._cache._entries.clear()
        # A late event re-rolls that day: the cache must not hide it.
        self.cc('old.jsonl', [self.cc_reply('o2', 'old', t + 10, 'ro2')])
        self.assertAlmostEqual(self.ledger(**q)['totals']['cost_usd'], first + 5.0)


class PiClassTest(TempDBTestCase):
    def test_account_class_from_org_type(self):
        from app.pi_lineage import account_class
        self.assertEqual(account_class('dotfleet_personal'), 'personal')
        self.assertEqual(account_class('dotfleet_work'), 'work')
        self.assertIsNone(account_class('claude_team'))
        self.assertIsNone(account_class(None))
