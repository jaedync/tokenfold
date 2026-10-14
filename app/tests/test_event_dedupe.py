"""Metadata-only Meridian/Pi replay through ingest, rollups and authenticated API."""
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from app.tests._support import TempDBTestCase


class MeridianPiAccountingTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        # Exercise the real read paths explicitly, not an unrelated background
        # cache worker that could outlive the temporary DB/pricing patches.
        rebuild = patch('app.aggregator.trigger_eager_rebuild')
        rebuild.start()
        self.addCleanup(rebuild.stop)
        from app.config import TZ_NAME
        self.midnight = datetime.now(ZoneInfo(TZ_NAME)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        self.now = self.midnight.timestamp() + 60
        self.before = (self.midnight - timedelta(seconds=1)).isoformat()
        self.after = (self.midnight + timedelta(seconds=1)).isoformat()
        self.days = [(self.midnight - timedelta(days=1)).strftime('%Y-%m-%d'),
                     self.midnight.strftime('%Y-%m-%d')]
        self.http = self.client()
        self.addCleanup(self.http.close)
        for name, value in (("DASHBOARD_USER", "fixture-user"),
                            ("DASHBOARD_PASSWORD", "fixture-password"),
                            ("LOCKED_SCOPE", None)):
            p = patch('app.config.' + name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch('app.api.time', Mock(wraps=time, time=lambda: self.now))
        p.start()
        self.addCleanup(p.stop)

    def pi(self, eid='pi', message_id='msg_shared', provider='Anthropic',
           cost=1.25, scope='personal'):
        return '/api/ingest/pi', {
            'machine': 'pi-machine', 'account_class': scope,
            'project_dir': '/metadata-only', 'session_file': eid + '.jsonl',
            'events': [{'event_id': eid, 'session_id': eid,
                        'timestamp': self.after, 'kind': 'assistant',
                        'provider': provider, 'api': 'anthropic-messages',
                        'model': 'claude-opus-4-8', 'request_id': message_id,
                        'usage': {'input': 1_000_000, 'cost_input': cost * .8,
                                  'cost_output': cost * .2, 'cost_total': cost}}]}

    def sdk(self, eid='sdk', message_id='msg_shared', scope='personal'):
        return '/api/ingest', {
            'machine': 'sdk-machine', 'project_dir': '/metadata-only',
            'session_file': eid + '.jsonl',
            'account_email': scope + '@example.test',
            'plan': 'enterprise' if scope == 'work' else 'personal',
            'events': [{'uuid': eid, 'sessionId': eid, 'type': 'assistant',
                        'timestamp': self.before, 'requestId': 'req_' + eid,
                        'message': {'id': message_id, 'model': 'claude-opus-4-8',
                                    'usage': {'input_tokens': 1_000_000},
                                    'content': []}}]}

    def post(self, payload, duplicate=False):
        route, body = payload
        response = self.http.post(route, json=body,
                                  headers={'X-API-Key': self.api_key})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['duplicates' if duplicate else 'accepted'],
                         len(body['events']))

    def api(self, scope):
        response = self.http.get('/api/rate-limits?scope=' + scope,
                                auth=('fixture-user', 'fixture-password'))
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['weekly_budget']

    def assert_costs(self, personal=1.25, work=0, other=0):
        from app.config import scope_predicate
        from app.cost_windows import compute_window_cost, compute_window_cost_by_model
        from app.aggregator import _build_dashboard_data_inner, _build_hourly
        for scope, expected in [('personal', personal), ('enterprise', work)]:
            pred = scope_predicate(scope)
            extra = other if scope == 'personal' else 0
            self.assertAlmostEqual(compute_window_cost(
                self.conn, self.now - 7 * 86400, self.now, scope=scope,
                anthropic_only=True), expected)
            models = compute_window_cost_by_model(
                self.conn, self.now - 7 * 86400, self.now, scope=scope,
                anthropic_only=True)
            self.assertAlmostEqual(sum(models.values()), expected)
            rows = self.conn.execute(
                f'SELECT cost, model_json FROM daily_summary WHERE {pred}').fetchall()
            self.assertAlmostEqual(sum(r['cost'] for r in rows), expected + extra)
            self.assertAlmostEqual(sum(m['cost'] for r in rows
                                       for m in json.loads(r['model_json']).values()),
                                   expected + extra)
            dashboard = _build_dashboard_data_inner(scope)
            self.assertAlmostEqual(sum(m['cost'] for m in dashboard['model_breakdown']),
                                   expected + extra)
            self.assertAlmostEqual(sum(h['cost'] for h in _build_hourly(self.conn, pred)),
                                   expected + extra)
            api = self.api(scope)
            self.assertEqual(api['week_cost'], round(expected, 2))
            self.assertAlmostEqual(sum(h['c'] for h in api['hourly_costs']), expected)

    def test_replay_arrival_orders_cross_midnight_and_authenticated_windows(self):
        self.assertEqual(self.http.get('/api/rate-limits?scope=personal').status_code, 401)
        for pi_first in (False, True):
            with self.subTest(pi_first=pi_first):
                self.reset()
                pair = [self.pi(), self.sdk()]
                if not pi_first:
                    pair.reverse()
                for payload in pair:
                    self.post(payload)
                for payload in pair:
                    self.post(payload, duplicate=True)
                self.assert_costs()
                by_day = dict(self.conn.execute('SELECT day, SUM(cost) FROM daily_summary GROUP BY day'))
                # The SDK row on days[0] is Meridian's twin: no cost and no
                # activity there, so that day has no rollup row at all.
                self.assertEqual(by_day, {self.days[1]: 1.25})
                from app.cost_windows import compute_window_cost
                boundary = self.midnight.timestamp()
                self.assertEqual(compute_window_cost(
                    self.conn, boundary - 10, boundary, scope='personal'), 0)
                self.assertEqual(compute_window_cost(
                    self.conn, boundary, boundary + 10, scope='personal'), 1.25)
                # Exact components survive canonicalization; no token-based repricing.
                raw = self.conn.execute("SELECT model_json FROM daily_summary WHERE account_email='pi-personal@dotfleet.local'").fetchone()
                model = next(iter(json.loads(raw[0]).values()))
                self.assertEqual(model['reported_cost_input'], 1)
                self.assertEqual(model['reported_cost_output'], .25)
                # Broader account window also includes direct Claude and another Pi session.
                self.post(self.sdk('direct', 'msg_direct'))
                self.post(self.pi('other-session', 'msg_other', cost=.75))
                self.post(self.pi('router', 'msg_shared', provider='openrouter', cost=2))
                self.post(self.pi('work-pi', 'msg_work', cost=3, scope='work'))
                self.post(self.sdk('work-sdk', 'msg_work', scope='work'))
                self.assert_costs(personal=7, work=3, other=2)
                from app.aggregator import _build_recent_sessions
                from app.config import PERSONAL_PRED
                sessions = _build_recent_sessions(self.conn, PERSONAL_PRED)
                pi_sid = self.conn.execute("SELECT session_id FROM events WHERE source_client='pi-agent' AND message_id='msg_shared' AND provider='Anthropic'").fetchone()[0]
                session = next(s for s in sessions if s['session_id'] == pi_sid)
                self.assertEqual(session['cost'], 1.25)
                self.assertNotEqual(session['cost'], self.api('personal')['week_cost'])
                reset = datetime.fromtimestamp(self.now + 3600, ZoneInfo('UTC')).isoformat()
                # A fresh managed observation, as the Meridian reporters store it;
                # window estimates are only served for a fresh observation.
                usage = {'data': {'five_hour': {'utilization': 10, 'resets_at': reset},
                                  'seven_day': {'utilization': 20, 'resets_at': reset},
                                  'seven_day_opus': {'utilization': 5, 'resets_at': reset}},
                         'source': 'meridian-oauth', 'observed_at_epoch': self.now + 60,
                         'updated_at': datetime.fromtimestamp(self.now + 60, ZoneInfo('UTC')).isoformat()}
                self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('oauth_usage',?)", (json.dumps(usage),))
                self.conn.commit()
                with patch('app.api.time.time', return_value=self.now + 120):
                    oauth = self.api('personal')['oauth']
                self.assertEqual(oauth['five_hour_window']['cost'], 7)
                self.assertEqual(oauth['limit_window']['cost'], 7)
                self.assertEqual(next(b for b in oauth['buckets'] if b['key'] == 'scoped:opus')['window_cost'], 7)

    def test_exact_ids_never_cross_work_personal_scope(self):
        self.post(self.sdk('personal', 'msg_personal'))
        self.post(self.pi('work', 'msg_personal', scope='work', cost=3))
        self.post(self.sdk('work', 'msg_work', scope='work'))
        self.post(self.pi('personal', 'msg_work', cost=1.25))
        self.assert_costs(personal=6.25, work=8)
        self.assertEqual(self.mirrors(), [])

    def test_non_anthropic_and_non_message_ids_do_not_suppress_direct_claude(self):
        self.post(self.sdk('router-sdk', 'msg_router'))
        self.post(self.pi('router-pi', 'msg_router', provider='openrouter', cost=2))
        self.post(self.sdk('non-message-sdk', 'not-a-message-id'))
        self.post(self.pi('non-message-pi', 'not-a-message-id'))
        self.assert_costs(personal=11.25, other=2)
        self.assertEqual(self.mirrors(), [])

    def test_configured_org_type_scope_and_zero_reported_cost(self):
        route, sdk = self.sdk(scope='personal')
        sdk['org_type'] = 'claude_enterprise'
        self.post((route, sdk))
        self.post(self.pi(scope='work', cost=0))
        # Same class through org_type; the $0 Pi row has tokens, so its priced
        # SDK twin is the one billed, once.
        self.assert_costs(personal=0, work=5)

    def test_only_positive_pi_cost_outranks_the_priced_sdk_twin(self):
        from app.cost_windows import compute_window_cost
        # Anthropic never serves tokens for free: an explicit $0 with tokens
        # means Pi had no rates for the model (claude-fable-5-1, 2026-09-02).
        self.post(self.sdk())
        self.post(self.pi(cost=0))
        self.assertEqual(compute_window_cost(
            self.conn, self.now - 100, self.now, scope='personal'), 5)
        self.post(self.sdk('sdk2', 'msg_paid'))
        self.post(self.pi('pi2', 'msg_paid', cost=1.25))
        self.assertEqual(compute_window_cost(
            self.conn, self.now - 100, self.now, scope='personal'), 6.25)

    def test_unbillable_pi_row_cannot_hide_claude(self):
        from app.cost_windows import compute_window_cost
        self.post(self.sdk())
        self.post(self.pi())
        for column, value in [('request_id', None), ('model', None),
                              ('model', '<synthetic>')]:
            with self.subTest(column=column, value=value):
                self.conn.execute('SAVEPOINT unbillable')
                self.conn.execute(f"UPDATE events SET {column}=? WHERE source_client='pi-agent'", (value,))
                self.assertEqual(compute_window_cost(
                    self.conn, self.now - 100, self.now, scope='personal'), 5)
                self.conn.execute('ROLLBACK TO unbillable')
                self.conn.execute('RELEASE unbillable')

    def test_missing_usage_preserves_sdk_and_token_only_pair_bills_once(self):
        from app.cost_windows import compute_window_cost
        cases = [(None, False, 5), (None, True, 5), ({}, False, 5),
                 ({'input': 1_000_000}, False, 5),
                 # An explicit $0 (aborted message, or a model without rates)
                 # is not cost evidence: the priced SDK twin is billed once.
                 ({'cost_total': 0}, False, 5), ({'cost_input': 0}, False, 5)]
        for usage, missing_model, expected in cases:
            for pi_first in (False, True):
                with self.subTest(usage=usage, missing_model=missing_model, pi_first=pi_first):
                    self.reset()
                    route, body = self.pi()
                    event = {k: v for k, v in body['events'][0].items()
                             if k != 'usage' and not (missing_model and k == 'model')}
                    if usage is not None:
                        event = {**event, 'usage': usage}
                    # Neither counterpart day is today: last arrival must reroll
                    # the other historic day, not accidentally pass via today's sweep.
                    event = {**event, 'timestamp':
                             (datetime.fromisoformat(self.after)-timedelta(days=3)).isoformat()}
                    pi = (route, {**body, 'events': [event]})
                    sdk_route, sdk_body = self.sdk()
                    sdk_event = {**sdk_body['events'][0], 'timestamp':
                                 (datetime.fromisoformat(self.before)-timedelta(days=3)).isoformat()}
                    sdk = (sdk_route, {**sdk_body, 'events': [sdk_event]})
                    for payload in ([pi, sdk] if pi_first else [sdk, pi]):
                        self.post(payload)
                    self.assertAlmostEqual(compute_window_cost(
                        self.conn, self.now-7*86400, self.now, scope='personal'), expected)
                    self.assertAlmostEqual(self.conn.execute(
                        'SELECT SUM(cost) FROM daily_summary').fetchone()[0], expected)
                    for payload in (pi, sdk):
                        self.post(payload, duplicate=True)
                    from app import db, main
                    self.conn.execute("DELETE FROM meta WHERE key='meridian_pi_dedupe_version'")
                    self.conn.commit()
                    db._migrate(self.conn)
                    main._backfill_if_needed()
                    self.assertAlmostEqual(self.conn.execute(
                        'SELECT SUM(cost) FROM daily_summary').fetchone()[0], expected)
                    self.assertAlmostEqual(compute_window_cost(
                        self.conn, self.now-7*86400, self.now, scope='personal'), expected)

    def test_zero_token_sdk_twin_of_unpriced_pi_row_counts_one_api_call(self):
        for pi_first in (False, True):
            with self.subTest(pi_first=pi_first):
                self.reset()
                route, body = self.sdk()
                event = {**body['events'][0], 'timestamp': self.after}
                event['message'] = {**event['message'], 'usage': {'input_tokens': 0}}
                sdk = (route, {**body, 'events': [event]})
                pi_route, pi_body = self.pi()
                pi_event = {**pi_body['events'][0], 'usage': {'input': 1_000_000}}
                pi = (pi_route, {**pi_body, 'events': [pi_event]})
                for payload in ([pi, sdk] if pi_first else [sdk, pi]):
                    self.post(payload)
                from app.event_dedupe import canonical_assistant_predicate
                survivors = self.conn.execute(
                    "SELECT source_client FROM events WHERE type='assistant' AND "
                    + canonical_assistant_predicate()).fetchall()
                self.assertEqual([r[0] for r in survivors], ['pi-agent'])

    def test_message_lookup_uses_indexes_and_bounded_batches(self):
        from app.event_dedupe import canonical_assistant_predicate, matching_claude_days
        self.post(self.sdk())
        self.post(self.pi())
        ids = {f'msg_absent_{i}' for i in range(1100)} | {'msg_shared'}
        self.assertEqual(matching_claude_days(self.conn, ids), {self.days[0]})
        plan = self.conn.execute(
            'EXPLAIN QUERY PLAN SELECT uuid FROM events WHERE type=\'assistant\' '
            'AND ' + canonical_assistant_predicate()).fetchall()
        details = '\n'.join(r['detail'] for r in plan)
        self.assertIn('SEARCH canonical_pi USING INDEX idx_events_pi_anthropic_message (message_id=?)', details)
        from app.event_dedupe import matching_pi_days
        queries = []
        self.conn.set_trace_callback(queries.append)
        try:
            self.assertEqual(matching_pi_days(self.conn, {'msg_shared'}), set())
        finally:
            self.conn.set_trace_callback(None)
        plan = self.conn.execute('EXPLAIN QUERY PLAN ' + queries[-1]).fetchall()
        self.assertTrue(any('SEARCH events USING INDEX idx_events_pi_anthropic_message'
                            in row['detail'] for row in plan))

    def _at(self, seconds):
        return (datetime.fromisoformat(self.after) + timedelta(seconds=seconds)).isoformat()

    def pi_session(self):
        events = [
            {'event_id': 'pi-u1', 'session_id': 'pi-s', 'timestamp': self._at(10),
             'kind': 'user', 'has_text': True},
            {'event_id': 'pi-a1', 'session_id': 'pi-s', 'timestamp': self._at(40),
             'kind': 'assistant', 'provider': 'Anthropic', 'api': 'anthropic-messages',
             'model': 'claude-opus-4-8', 'request_id': 'msg_turn', 'has_tool_use': True,
             'tools': [{'tool_use_id': 'toolu_pi', 'name': 'bash'}],
             'usage': {'input': 1_000_000, 'cost_input': 1.0, 'cost_output': .25,
                       'cost_total': 1.25}},
            {'event_id': 'pi-r1', 'session_id': 'pi-s', 'timestamp': self._at(70),
             'kind': 'tool_result', 'has_tool_result': True}]
        return '/api/ingest/pi', {
            'machine': 'pi-machine', 'account_class': 'personal',
            'project_dir': '/metadata-only', 'session_file': 'pi-s.jsonl', 'events': events}

    def sdk_session(self, sid, message_ids):
        """A Claude Code transcript: one prompt, then a tool turn per message ID."""
        recs = [{'uuid': sid + '-u', 'sessionId': sid, 'type': 'user', 'userType': 'external',
                 'timestamp': self._at(11),
                 'message': {'role': 'user', 'content': 'run the tests'}}]
        for i, mid in enumerate(message_ids):
            t = 41 + 60 * i
            recs.append({'uuid': f'{sid}-a{i}', 'sessionId': sid, 'type': 'assistant',
                         'timestamp': self._at(t), 'requestId': f'req_{sid}_{i}',
                         'message': {'id': mid, 'model': 'claude-opus-4-8',
                                     'usage': {'input_tokens': 1_000_000},
                                     'content': [{'type': 'tool_use', 'id': f'toolu_{sid}_{i}',
                                                  'name': 'bash', 'input': {}}]}})
            recs.append({'uuid': f'{sid}-r{i}', 'sessionId': sid, 'type': 'user',
                         'userType': 'external', 'timestamp': self._at(t + 30),
                         'message': {'role': 'user', 'content': [
                             {'type': 'tool_result', 'tool_use_id': f'toolu_{sid}_{i}',
                              'content': 'ok'}]}})
        return '/api/ingest', {
            'machine': 'sdk-machine', 'project_dir': '/metadata-only',
            'session_file': sid + '.jsonl', 'account_email': 'personal@example.test',
            'plan': 'personal', 'events': recs}

    def activity(self):
        from app.aggregator import _build_hourly, _build_recent_sessions
        from app.api import _active_seconds
        from app.config import PERSONAL_PRED
        row = self.conn.execute(
            'SELECT SUM(sessions), SUM(human_prompts), SUM(tool_calls), '
            'ROUND(SUM(active_s), 3), ROUND(SUM(cost), 4) FROM daily_summary').fetchone()
        hourly = _build_hourly(self.conn, PERSONAL_PRED)
        return {'rollup': tuple(row[:4]), 'cost': row[4],
                'hourly': (sum(h.get('prompts', 0) for h in hourly),
                           sum(h.get('tool_calls', 0) for h in hourly)),
                'active': round(_active_seconds(self.conn, PERSONAL_PRED,
                                                self.now - 86400, self.now + 86400), 3),
                'sessions': sorted(s['session_id'] for s in
                                   _build_recent_sessions(self.conn, PERSONAL_PRED))}

    def reset(self):
        for table in ('events', 'daily_summary', 'sync_cursors', 'tool_uses',
                      'pi_mirror_sessions'):
            self.conn.execute('DELETE FROM ' + table)
        self.conn.commit()

    def test_meridian_twin_session_is_not_counted_as_activity(self):
        from app import db, main
        direct = self.sdk_session('sdk-direct', ['msg_direct'])
        # The SDK copy of the Pi session, plus one internal Meridian call
        # with no Pi twin: real spend that must still bill once.
        mirror = self.sdk_session('sdk-mirror', ['msg_turn', 'msg_internal'])
        self.reset()
        for payload in (self.pi_session(), direct):
            self.post(payload)
        baseline = self.activity()
        self.assertEqual(baseline['rollup'][:3], (2, 2, 2))
        for mirror_first in (False, True):
            with self.subTest(mirror_first=mirror_first):
                self.reset()
                order = ([mirror, self.pi_session(), direct] if mirror_first
                         else [self.pi_session(), direct, mirror])
                for payload in order:
                    self.post(payload)
                self.assertEqual([r[0] for r in self.conn.execute(
                    'SELECT session_id FROM pi_mirror_sessions')], ['sdk-mirror'])
                got = self.activity()
                self.assertEqual({k: v for k, v in got.items() if k != 'cost'},
                                 {k: v for k, v in baseline.items() if k != 'cost'})
                self.assertAlmostEqual(got['cost'], baseline['cost'] + 5)
        # The migration rebuilds the twin set from history alone.
        self.conn.execute('DELETE FROM pi_mirror_sessions')
        self.conn.execute("DELETE FROM meta WHERE key='meridian_pi_dedupe_version'")
        self.conn.commit()
        db._migrate(self.conn)
        main._backfill_if_needed()
        self.assertEqual([r[0] for r in self.conn.execute(
            'SELECT session_id FROM pi_mirror_sessions')], ['sdk-mirror'])
        self.assertEqual({k: v for k, v in self.activity().items() if k != 'cost'},
                         {k: v for k, v in baseline.items() if k != 'cost'})

    def mirrors(self):
        return [r[0] for r in self.conn.execute(
            'SELECT session_id FROM pi_mirror_sessions ORDER BY session_id')]

    def test_sdk_session_without_reply_is_not_activity(self):
        # Meridian's SDK copy is often caught before its reply is written and
        # then deleted: only the replayed prompt, attachments and tool result
        # remain, while Pi already counts that turn.
        self.reset()
        for payload in (self.pi_session(), self.sdk_session('sdk-direct', ['msg_direct'])):
            self.post(payload)
        baseline = self.activity()
        route, body = self.sdk_session('sdk-partial', ['msg_never_ingested'])
        prompt, _, result = body['events']
        attachment = {'uuid': 'sdk-partial-att', 'sessionId': 'sdk-partial',
                      'type': 'attachment', 'timestamp': self._at(12),
                      'attachment': {'type': 'environment'}}
        self.post((route, {**body, 'events': [prompt, attachment,
                                              {**result, 'timestamp': self._at(14)}]}))
        self.assertEqual(self.activity(), baseline)
        # A direct session counts from its first reply, its prompt included.
        route, body = self.sdk_session('sdk-late', ['msg_late'])
        self.post((route, {**body, 'events': body['events'][:1]}))
        self.assertEqual(self.activity(), baseline)
        self.post((route, {**body, 'events': body['events'][1:]}))
        self.assertEqual(self.activity()['rollup'][:3], (3, 3, 3))

    def test_first_reply_rerolls_earlier_days_and_pi_is_unaffected(self):
        self.reset()
        route, body = self.sdk_session('sdk-late', ['msg_late'])
        prompt = {**body['events'][0], 'timestamp': self.before}
        self.post((route, {**body, 'events': [prompt]}))
        def first_day():
            return tuple(self.conn.execute(
                'SELECT SUM(sessions), SUM(human_prompts) FROM daily_summary '
                'WHERE day=?', (self.days[0],)).fetchone())
        self.assertEqual(first_day(), (None, None))
        # The first reply lands after midnight: the prompt's day now counts.
        self.post((route, {**body, 'events': body['events'][1:]}))
        self.assertEqual(first_day(), (1, 1))
        # Pi sessions never need a reply to count. A '<synthetic>' row is not
        # a reply: Claude Code writes it after an API error, with no model call.
        pi_route, pi_body = self.pi_session()
        self.post((pi_route, {**pi_body, 'events': pi_body['events'][:1]}))
        synthetic = self.sdk_session('sdk-synthetic', ['msg_synthetic'])
        events = synthetic[1]['events'][:2]
        events[1] = {**events[1], 'message': {**events[1]['message'], 'model': '<synthetic>',
                                              'usage': {}, 'content': []}}
        self.post((synthetic[0], {**synthetic[1], 'events': events}))
        self.assertEqual(self.activity()['rollup'][:2], (3, 2))

    def test_mirror_scan_bounds_combine(self):
        from app.event_dedupe import find_mirror_sessions
        self.reset()
        self.post(self.pi_session())
        self.post(self.sdk_session('sdk-mirror', ['msg_turn']))
        self.conn.execute('DELETE FROM pi_mirror_sessions')
        self.conn.commit()
        for since, expected in ((self.now + 86400, set()), (self.now - 86400, {'sdk-mirror'})):
            self.assertEqual(find_mirror_sessions(
                self.conn, {'msg_turn', 'msg_other'}, since_epoch=since), expected)
        self.assertEqual(find_mirror_sessions(self.conn, {'msg_other'},
                                              since_epoch=self.now - 86400), set())

    def test_twin_only_machine_is_not_last_active(self):
        from app.aggregator import _build_dashboard_data_inner
        self.reset()
        self.post(self.pi_session())
        # The SDK copy ends after Pi's last event, on its own Meridian host.
        self.post(self.sdk_session('sdk-mirror', ['msg_turn', 'msg_internal']))
        self.assertEqual(self.mirrors(), ['sdk-mirror'])
        data = _build_dashboard_data_inner('personal')
        pi_last = self.conn.execute(
            "SELECT MAX(ts_epoch) FROM events WHERE source_client='pi-agent'").fetchone()[0]
        self.assertEqual(set(data['machine_last_active']), {'pi-machine'})
        self.assertEqual(data['last_active_ts'], pi_last)

    def test_enterprise_session_without_reply_is_not_activity_and_cost_is_unchanged(self):
        from app.config import scope_predicate
        self.reset()

        def enterprise(sid, message_ids, events=None):
            route, body = self.sdk_session(sid, message_ids)
            return route, {**body, 'account_email': 'work@example.test',
                           'plan': 'enterprise',
                           'events': body['events'] if events is None else events}

        def rollup():
            row = self.conn.execute(
                'SELECT SUM(sessions), SUM(human_prompts), SUM(cost) FROM daily_summary '
                'WHERE ' + scope_predicate('enterprise')).fetchone()
            return tuple(row)

        self.post(enterprise('ent-real', ['msg_ent']))
        baseline = rollup()
        self.assertEqual(baseline[:2], (1, 1))
        self.assertAlmostEqual(baseline[2], 5.0)
        # Prompt and tool result only: the model never replied.
        route, body = enterprise('ent-silent', ['msg_none'])
        prompt, _, result = body['events']
        self.post((route, {**body, 'events': [prompt, result]}))
        after = rollup()
        self.assertEqual(after[:2], (1, 1))
        self.assertAlmostEqual(after[2], baseline[2])

    def test_recent_pi_message_ids_use_time_index_and_bound_the_sweep(self):
        from app.event_dedupe import RECENT_PI_MESSAGES_SQL, recent_pi_message_ids
        self.reset()
        self.post(self.pi_session())
        plan = '\n'.join(r['detail'] for r in self.conn.execute(
            'EXPLAIN QUERY PLAN ' + RECENT_PI_MESSAGES_SQL, (0,)))
        self.assertIn('USING INDEX idx_events_billable_ts (ts_epoch>?)', plan)
        self.assertEqual(recent_pi_message_ids(self.conn, self.now - 86400), {'msg_turn'})
        self.assertEqual(recent_pi_message_ids(self.conn, self.now + 86400), set())

    def test_hourly_sweep_checks_only_recent_pi_message_ids(self):
        from app import aggregator, event_dedupe
        self.reset()
        self.post(self.pi_session())
        self.post(self.sdk_session('sdk-mirror', ['msg_turn']))
        self.conn.execute('DELETE FROM pi_mirror_sessions')
        self.conn.commit()
        calls = []
        real = event_dedupe.find_mirror_sessions

        def spy(conn, ids=None, **kw):
            calls.append((ids, kw))
            return real(conn, ids, **kw)

        with patch.object(aggregator, '_schedule_periodic_sweep'), \
                patch.object(aggregator, '_last_full_sweep', 0.0), \
                patch('app.event_dedupe.find_mirror_sessions', spy):
            aggregator._run_periodic_sweep()
        self.assertEqual(self.mirrors(), ['sdk-mirror'])
        # Never an unbounded all-history or SDK-side time scan.
        self.assertEqual(calls, [({'msg_turn'}, {'since_epoch': None})])

    def test_twin_found_later_rerolls_earlier_days_of_the_session(self):
        self.reset()
        self.post(self.pi_session())
        route, body = self.sdk_session('sdk-mirror', ['msg_internal', 'msg_turn'])
        def day_before(event):
            ts = datetime.fromisoformat(event['timestamp']) - timedelta(days=1)
            return {**event, 'timestamp': ts.isoformat()}
        def first_day():
            return tuple(self.conn.execute(
                'SELECT SUM(sessions), SUM(human_prompts), SUM(tool_calls), SUM(cost) '
                'FROM daily_summary WHERE day=?', (self.days[0],)).fetchone())
        # The previous day holds the prompt and Meridian's own call; nothing
        # there shares an ID with Pi, so the session is not a twin yet.
        self.post((route, {**body, 'events': [day_before(e) for e in body['events'][:3]]}))
        self.assertEqual(first_day(), (1, 1, 1, 5))
        self.post((route, {**body, 'events': body['events'][3:]}))
        self.assertEqual(self.mirrors(), ['sdk-mirror'])
        self.assertEqual(first_day(), (0, 0, 0, 5))

    def test_failed_mark_is_repaired_by_resend_or_hourly_sweep(self):
        from app import aggregator
        for repair in ('resend', 'sweep'):
            with self.subTest(repair=repair):
                self.reset()
                self.post(self.pi_session())
                mirror = self.sdk_session('sdk-mirror', ['msg_turn'])
                with patch('app.event_dedupe.find_mirror_sessions',
                           side_effect=RuntimeError('simulated')), \
                        self.assertLogs('app.event_dedupe', level='ERROR'):
                    self.post(mirror)
                self.assertEqual(self.mirrors(), [])
                if repair == 'resend':
                    # A crash after commit: the client resends; every row is
                    # a duplicate, but the mark still runs.
                    self.post(mirror, duplicate=True)
                else:
                    # A logged failure returned 200, so nothing resends.
                    with patch.object(aggregator, '_schedule_periodic_sweep'), \
                            patch.object(aggregator, '_last_full_sweep', 0.0):
                        aggregator._run_periodic_sweep()
                self.assertEqual(self.mirrors(), ['sdk-mirror'])
                self.assertEqual(self.activity()['rollup'][:3], (1, 1, 1))

    def test_reroll_lookups_use_partial_indexes_with_bound_parameters(self):
        from app.event_dedupe import matching_claude_days, matching_pi_days
        self.post(self.sdk())
        self.post(self.pi())
        for lookup, index in ((matching_claude_days, 'idx_events_claude_message'),
                              (matching_pi_days, 'idx_events_pi_anthropic_message')):
            calls = []

            class Recorder:
                def execute(inner, sql, params=()):
                    calls.append((sql, list(params)))
                    return self.conn.execute(sql, params)
            lookup(Recorder(), {'msg_shared', 'msg_other'})
            sql, params = calls[0]
            plan = '\n'.join(r['detail'] for r in self.conn.execute(
                'EXPLAIN QUERY PLAN ' + sql, params))
            self.assertIn(f'SEARCH events USING INDEX {index}', plan, lookup.__name__)

    def test_historical_migration_populated_ids_and_interrupted_rebuild(self):
        from app import db, main, summarizer
        self.post(self.sdk())
        self.post(self.pi())
        self.post(self.sdk('direct', 'msg_direct'))
        # Emulate pre-release rollups with IDs already populated. No ID UPDATE
        # rowcount can signal this stale-accounting case.
        with patch('app.summarizer.canonical_assistant_predicate', return_value='1'):
            summarizer.summarize_days(self.days)
        self.conn.execute("DELETE FROM meta WHERE key='meridian_pi_dedupe_version'")
        self.conn.execute(
            "INSERT INTO billing_readings(amount_usd,month,recorded_at,recorded_epoch) "
            "VALUES(17,'2026-09','fixture',?)", (self.now,))
        self.conn.execute("INSERT INTO meta(key,value) VALUES('preserve_fixture','untouched')")
        self.conn.commit()
        db._migrate(self.conn)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM daily_summary').fetchone()[0], 0)
        real = summarizer.summarize_days
        def interrupt(days):
            if days == [self.days[0]]:
                raise RuntimeError('simulated interrupted backfill')
            real(days)
        # A failing day must not crash-loop startup: log it, finish the other
        # days, keep the durable marker, and serve.
        with patch('app.summarizer.summarize_days', side_effect=interrupt), \
                self.assertLogs('tokenfold.main', level='ERROR') as logs:
            main._backfill_if_needed()
        self.assertIn(self.days[0], '\n'.join(logs.output))
        self.assertEqual(self.conn.execute(
            "SELECT value FROM meta WHERE key='daily_summary_rebuild_pending'").fetchone()[0], '1')
        # The day after the failure was still rebuilt; the failed day was not.
        self.assertEqual({r[0] for r in self.conn.execute('SELECT DISTINCT day FROM daily_summary')},
                         {self.days[1]})
        db.close_conn()
        self.conn = db.get_conn()
        main._backfill_if_needed()
        self.assertEqual({r[0] for r in self.conn.execute('SELECT DISTINCT day FROM daily_summary')},
                         set(self.days))
        self.assert_costs(personal=6.25)
        rows = [tuple(r) for r in self.conn.execute('SELECT * FROM daily_summary ORDER BY day,account_email')]
        db._migrate(self.conn)
        main._backfill_if_needed()
        self.assertEqual(rows, [tuple(r) for r in self.conn.execute('SELECT * FROM daily_summary ORDER BY day,account_email')])
        self.assertEqual(self.conn.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        self.assertEqual(self.conn.execute('SELECT amount_usd FROM billing_readings').fetchone()[0], 17)
        self.assertEqual(self.conn.execute("SELECT value FROM meta WHERE key='preserve_fixture'").fetchone()[0], 'untouched')
        self.assertIsNone(self.conn.execute("SELECT value FROM meta WHERE key='daily_summary_rebuild_pending'").fetchone())

    def test_historical_backfill_validates_namespace_and_preserves_raw_rows(self):
        from app import db, main
        self.post(self.sdk())
        self.post(self.pi())
        self.post(self.pi('malformed', 'msg_malformed', cost=.75))
        self.conn.execute("UPDATE events SET message_id=NULL WHERE source_client='pi-agent'")
        self.conn.execute("UPDATE events SET request_id=? WHERE request_id LIKE '%:msg_malformed'", ('pi:' + 'z' * 40 + ':msg_malformed',))
        self.conn.commit()
        before = self.conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        db._migrate(self.conn)
        self.assertIsNone(self.conn.execute("SELECT message_id FROM events WHERE request_id LIKE '%:msg_malformed'").fetchone()[0])
        main._backfill_if_needed()
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM events').fetchone()[0], before)
        self.assert_costs(personal=2)
