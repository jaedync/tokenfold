"""A Pi session file copied to another machine (or forked) is one set of requests.

The Pi namespace hashes the machine and session file, so a copied file gets
new uuids, session ids and request ids, and the raw event ids can differ too
(the client hashes the file path). The provider response ID and the entry
content stay the same. The first-ingested row is kept; later copies are
flagged at ingest (or by the migration) and excluded from cost and activity.
"""
import itertools
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from app.tests._support import TempDBTestCase
from app.tests.test_ha_windows import _HABase


def _iso(epoch):
    return datetime.fromtimestamp(epoch, ZoneInfo('UTC')).isoformat()


def pi_events(t0, ids='', native='native-s1', anthropic_cost=1.25,
              is_sidechain=False):
    """One Pi turn: prompt, Anthropic reply with a tool, tool result, a Codex
    reply and a compaction with no response ID (event: fallback)."""
    def at(s):
        return _iso(t0 + s)
    usage = {'input': 1_000_000, 'output': 1000}
    if anthropic_cost is not None:
        usage = {**usage, 'cost_input': anthropic_cost * .8,
                 'cost_output': anthropic_cost * .2, 'cost_total': anthropic_cost}
    common = {'session_id': native, 'is_sidechain': is_sidechain}
    return [
        {**common, 'event_id': ids + 'u1', 'timestamp': at(10), 'kind': 'user',
         'has_text': True, 'text_length': 42},
        {**common, 'event_id': ids + 'a1', 'timestamp': at(40), 'kind': 'assistant',
         'provider': 'anthropic', 'api': 'anthropic-messages',
         'model': 'claude-opus-4-8', 'request_id': 'msg_a1', 'has_tool_use': True,
         'tools': [{'tool_use_id': 'toolu_1', 'name': 'bash'}], 'usage': usage},
        {**common, 'event_id': ids + 'r1', 'timestamp': at(70), 'kind': 'tool_result',
         'has_tool_result': True, 'text_length': 500},
        {**common, 'event_id': ids + 'c1', 'timestamp': at(100), 'kind': 'assistant',
         'provider': 'openai-codex', 'api': 'openai-codex-responses',
         'model': 'gpt-6', 'request_id': 'resp_c1',
         'usage': {'input': 2000, 'output': 300, 'cost_total': .5}},
        {**common, 'event_id': ids + 'k1', 'timestamp': at(130), 'kind': 'compaction',
         'provider': 'anthropic', 'api': 'anthropic-messages',
         'model': 'claude-opus-4-8', 'usage': {'input': 5000, 'cost_total': .1}},
    ]


class _PiCopyBase(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        rebuild = patch('app.aggregator.trigger_eager_rebuild')
        rebuild.start()
        self.addCleanup(rebuild.stop)
        from app.config import TZ_NAME
        midnight = datetime.now(ZoneInfo(TZ_NAME)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        self.t0 = midnight.timestamp() + 60
        self.now = self.t0 + 3600
        self.http = self.client()
        self.addCleanup(self.http.close)
        p = patch('app.api.time', Mock(wraps=time, time=lambda: self.now))
        p.start()
        self.addCleanup(p.stop)

    def pi(self, machine, session_file, events, scope='personal'):
        return '/api/ingest/pi', {
            'machine': machine, 'account_class': scope, 'project_dir': '/' + machine,
            'session_file': session_file, 'events': events}

    def sdk(self, sid='sdk-mirror', message_id='msg_a1'):
        """Meridian's Claude Code copy of the Anthropic reply (priced $5)."""
        def at(s):
            return _iso(self.t0 + s)
        return '/api/ingest', {
            'machine': 'meridian-host', 'project_dir': '/meridian',
            'session_file': sid + '.jsonl', 'account_email': 'personal@example.test',
            'plan': 'personal', 'events': [
                {'uuid': sid + '-u', 'sessionId': sid, 'type': 'user',
                 'userType': 'external', 'timestamp': at(11),
                 'message': {'role': 'user', 'content': 'run the tests'}},
                {'uuid': sid + '-a', 'sessionId': sid, 'type': 'assistant',
                 'timestamp': at(41), 'requestId': 'req_' + sid,
                 'message': {'id': message_id, 'model': 'claude-opus-4-8',
                             'usage': {'input_tokens': 1_000_000}, 'content': []}}]}

    def post(self, payload, duplicate=False):
        route, body = payload
        response = self.http.post(route, json=body, headers={'X-API-Key': self.api_key})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['duplicates' if duplicate else 'accepted'],
                         len(body['events']))

    def reset(self):
        for table in ('events', 'daily_summary', 'sync_cursors', 'tool_uses',
                      'pi_mirror_sessions'):
            self.conn.execute('DELETE FROM ' + table)
        self.conn.commit()

    def metrics(self, scope='personal'):
        from app.aggregator import (_build_dashboard_data_inner, _build_hourly,
                                    _build_recent_sessions)
        from app.api import _active_seconds
        from app.config import scope_predicate
        from app.cost_windows import compute_window_cost
        from app.provider_usage import _month_reported_costs, _window_reported_cost
        pred = scope_predicate(scope)
        start, end = self.t0 - 86400, self.t0 + 86400
        rollup = self.conn.execute(
            'SELECT SUM(sessions), SUM(human_prompts), SUM(tool_calls), '
            f'ROUND(SUM(active_s), 3), ROUND(SUM(cost), 4) FROM daily_summary WHERE {pred}'
        ).fetchone()
        models = self.conn.execute(
            f'SELECT model_json FROM daily_summary WHERE {pred}').fetchall()
        api_calls = sum(m['api_calls'] for r in models
                        for m in json.loads(r[0]).values())
        hourly = _build_hourly(self.conn, pred)
        dashboard = _build_dashboard_data_inner(scope)
        with patch('time.time', return_value=self.now):
            month = _month_reported_costs(self.conn, self.now, scope)
        return {
            'cost': round(compute_window_cost(self.conn, start, end, scope=scope), 4),
            'claude_cost': round(compute_window_cost(
                self.conn, start, end, scope=scope, anthropic_only=True), 4),
            'codex_window': round(_window_reported_cost(
                self.conn, scope, 'codex', start, end) or 0, 4),
            'codex_month': month['codex'],
            'rollup': tuple(rollup), 'api_calls': api_calls,
            'hourly': (round(sum(h['cost'] for h in hourly), 4),
                       sum(h.get('prompts', 0) for h in hourly),
                       sum(h.get('tool_calls', 0) for h in hourly)),
            'dashboard_cost': round(sum(m['cost'] for m in dashboard['model_breakdown']), 4),
            'active': round(_active_seconds(self.conn, pred, start, end), 3),
            'sessions': len(_build_recent_sessions(self.conn, pred)),
        }

    def survivors(self, message_id='msg_a1'):
        """Machines of the rows that bill this response."""
        from app.event_dedupe import canonical_assistant_predicate
        return sorted(r[0] for r in self.conn.execute(
            "SELECT source_machine FROM events WHERE type='assistant' "
            f"AND message_id=? AND {canonical_assistant_predicate()}", (message_id,)))

    def baseline(self, account_class='personal'):
        self.reset()
        self.post(self.pi('orig', 'proj-a/s1.jsonl', pi_events(self.t0),
                          scope=account_class))
        got = self.metrics('enterprise' if account_class == 'work' else 'personal')
        self.reset()
        return got


class PiCopyTest(_PiCopyBase):
    def test_baseline_counts_each_part_of_the_turn(self):
        base = self.baseline()
        # $1.25 Anthropic + $0.50 Codex + $0.10 compaction, one session,
        # one prompt, one tool call, three billed requests.
        self.assertEqual(base['cost'], 1.85)
        self.assertEqual(base['claude_cost'], 1.35)
        self.assertEqual(base['codex_window'], .5)
        self.assertEqual(base['rollup'][:3], (1, 1, 1))
        self.assertEqual(base['api_calls'], 3)
        self.assertEqual(base['sessions'], 1)

    def test_copy_on_another_machine_counts_once_in_both_orders(self):
        base = self.baseline()
        orig = self.pi('orig', 'proj-a/s1.jsonl', pi_events(self.t0))
        for same_ids, copy_first in itertools.product((True, False), (False, True)):
            with self.subTest(same_ids=same_ids, copy_first=copy_first):
                self.reset()
                # A different home directory changes the client's event IDs.
                copy = self.pi('copy', 'proj-b/s1.jsonl', pi_events(
                    self.t0, ids='' if same_ids else 'other-path-'))
                order = [copy, orig] if copy_first else [orig, copy]
                for payload in order:
                    self.post(payload)
                for payload in order:
                    self.post(payload, duplicate=True)
                self.assertEqual(self.metrics(), base)
                first = 'copy' if copy_first else 'orig'
                self.assertEqual(self.survivors(), [first])
                flagged = {r[0] for r in self.conn.execute(
                    'SELECT DISTINCT source_machine FROM events WHERE pi_copy=1')}
                self.assertEqual(flagged, {'orig', 'copy'} - {first})
                self.assertEqual(self.conn.execute(
                    'SELECT COUNT(*) FROM tool_uses WHERE pi_copy=1').fetchone()[0], 1)

    def test_three_copies_keep_only_the_first(self):
        base = self.baseline()
        for i, machine in enumerate(('orig', 'copy-1', 'copy-2')):
            self.post(self.pi(machine, f'proj-{i}/s1.jsonl',
                              pi_events(self.t0, ids=f'path{i}-')))
        self.assertEqual(self.metrics(), base)
        self.assertEqual(self.survivors(), ['orig'])
        self.assertEqual(self.conn.execute(
            'SELECT COUNT(*) FROM events WHERE pi_copy=1').fetchone()[0], 2 * 5)

    def test_work_and_personal_copies_never_merge(self):
        base, work = self.baseline(), self.baseline('work')
        self.post(self.pi('orig', 'proj-a/s1.jsonl', pi_events(self.t0)))
        self.post(self.pi('copy', 'proj-b/s1.jsonl', pi_events(self.t0), scope='work'))
        self.assertEqual(self.metrics('personal'), base)
        self.assertEqual(self.metrics('enterprise'), work)
        self.assertEqual(work['cost'], 1.85)
        self.assertEqual(self.conn.execute(
            'SELECT COUNT(*) FROM events WHERE pi_copy=1').fetchone()[0], 0)

    def test_copy_with_meridian_twin_bills_once_in_every_order(self):
        base = self.baseline()
        orig = self.pi('orig', 'proj-a/s1.jsonl', pi_events(self.t0))
        copy = self.pi('copy', 'proj-b/s1.jsonl', pi_events(self.t0, ids='x-'))
        for order in itertools.permutations((orig, copy, self.sdk())):
            with self.subTest(order=[body['machine'] for _, body in order]):
                self.reset()
                for payload in order:
                    self.post(payload)
                first_pi = next(b['machine'] for _, b in order if b['machine'] != 'meridian-host')
                self.assertEqual(self.survivors(), [first_pi])
                self.assertEqual(self.conn.execute(
                    'SELECT session_id FROM pi_mirror_sessions').fetchall()[0][0], 'sdk-mirror')
                self.assertEqual(self.metrics(), base)

    def test_unpriced_copies_yield_to_the_twin_and_never_vanish(self):
        """Pi without cost yields to the priced SDK row. The request keeps
        exactly one row even when only the later copy reports a cost."""
        for orig_cost, copy_cost in ((None, None), (None, 1.25)):
            for order in itertools.permutations(('orig', 'copy', 'sdk')):
                with self.subTest(orig_cost=orig_cost, copy_cost=copy_cost, order=order):
                    self.reset()
                    payloads = {
                        'orig': self.pi('orig', 'proj-a/s1.jsonl',
                                        pi_events(self.t0, anthropic_cost=orig_cost)),
                        'copy': self.pi('copy', 'proj-b/s1.jsonl',
                                        pi_events(self.t0, ids='x-', anthropic_cost=copy_cost)),
                        'sdk': self.sdk()}
                    for name in order:
                        self.post(payloads[name])
                    first_pi = next(n for n in order if n != 'sdk')
                    first_cost = orig_cost if first_pi == 'orig' else copy_cost
                    self.assertEqual(self.survivors(),
                                     ['meridian-host'] if first_cost is None else [first_pi])
                    expected = (5 if first_cost is None else first_cost) + .6
                    self.assertAlmostEqual(self.metrics()['cost'], expected)
                    self.assertAlmostEqual(self.metrics()['rollup'][4], expected)

    def test_copy_continued_on_the_new_machine_counts_only_new_rows(self):
        base = self.baseline()
        self.post(self.pi('orig', 'proj-a/s1.jsonl', pi_events(self.t0)))
        later = [{'session_id': 'native-s1', 'event_id': 'x-u2', 'kind': 'user',
                  'timestamp': _iso(self.t0 + 200), 'has_text': True, 'text_length': 7},
                 {'session_id': 'native-s1', 'event_id': 'x-a2', 'kind': 'assistant',
                  'timestamp': _iso(self.t0 + 230), 'provider': 'anthropic',
                  'model': 'claude-opus-4-8', 'request_id': 'msg_a2',
                  'usage': {'input': 10, 'cost_total': 2.0}}]
        self.post(self.pi('copy', 'proj-b/s1.jsonl', pi_events(self.t0, ids='x-') + later))
        got = self.metrics()
        self.assertAlmostEqual(got['cost'], base['cost'] + 2)
        self.assertEqual(got['rollup'][1:3], (base['rollup'][1] + 1, base['rollup'][2]))
        self.assertEqual(got['api_calls'], base['api_calls'] + 1)

    def test_copy_continued_after_midnight_starts_no_gap_at_a_copy_row(self):
        """A one-day rebuild seeds active time from the previous day's last
        row of each session. A copy row is not activity there either."""
        from app import summarizer
        midnight = self.t0 - 60
        self.post(self.pi('orig', 'proj-a/s1.jsonl', pi_events(midnight - 200)))
        later = [{'session_id': 'native-s1', 'event_id': 'x-u2', 'kind': 'user',
                  'timestamp': _iso(midnight + 20), 'has_text': True, 'text_length': 7},
                 {'session_id': 'native-s1', 'event_id': 'x-a2', 'kind': 'assistant',
                  'timestamp': _iso(midnight + 50), 'provider': 'anthropic',
                  'model': 'claude-opus-4-8', 'request_id': 'msg_a2',
                  'usage': {'input': 10, 'cost_total': 2.0}}]
        self.post(self.pi('copy', 'proj-b/s1.jsonl',
                          pi_events(midnight - 200, ids='x-') + later))
        day = datetime.fromtimestamp(midnight + 20, ZoneInfo(summarizer.TZ_NAME)).strftime('%Y-%m-%d')

        def active_on_day():
            return self.conn.execute(
                'SELECT ROUND(SUM(active_s), 3) FROM daily_summary WHERE day=?',
                (day,)).fetchone()[0]
        both_days = active_on_day()
        self.assertEqual(both_days, 30.0)  # the new prompt to its reply only
        summarizer.summarize_days([day])
        self.assertEqual(active_on_day(), both_days)

    def test_fork_on_the_same_machine_counts_copied_history_once(self):
        """Pi /fork writes the history into a new session with a new native
        session ID, keeping each entry's timestamp and response ID."""
        base = self.baseline()
        self.post(self.pi('orig', 'proj-a/s1.jsonl', pi_events(self.t0)))
        fork = pi_events(self.t0, ids='fork-', native='native-s2')
        fork.append({'session_id': 'native-s2', 'event_id': 'fork-u2', 'kind': 'user',
                     'timestamp': _iso(self.t0 + 300), 'has_text': True, 'text_length': 9})
        self.post(self.pi('orig', 'proj-a/s2.jsonl', fork))
        got = self.metrics()
        self.assertEqual(got['cost'], base['cost'])
        self.assertEqual(got['rollup'][:3], (2, base['rollup'][1] + 1, base['rollup'][2]))
        self.assertEqual(got['api_calls'], base['api_calls'])

    def test_unrelated_sessions_with_equal_content_do_not_merge(self):
        # Subagent (sidechain) rows match only inside one native session, and
        # different response IDs are always different requests.
        sub_a = pi_events(self.t0, ids='a-', native='child-a', is_sidechain=True)
        sub_b = pi_events(self.t0, ids='b-', native='child-b', is_sidechain=True)
        for event in sub_b:
            if event.get('request_id') == 'msg_a1':
                event['request_id'] = 'msg_b1'
            if event.get('request_id') == 'resp_c1':
                event['request_id'] = 'resp_c2'
        self.post(self.pi('orig', 'sub/a.jsonl', sub_a))
        self.post(self.pi('orig', 'sub/b.jsonl', sub_b))
        self.assertEqual(self.conn.execute(
            'SELECT COUNT(*) FROM events WHERE pi_copy=1').fetchone()[0], 0)
        self.assertAlmostEqual(self.metrics()['cost'], 2 * 1.85)

    def test_flag_lookup_uses_the_partial_key_index(self):
        from app.event_dedupe import pi_copy_flag_sql
        plan = '\n'.join(r['detail'] for r in self.conn.execute(
            'EXPLAIN QUERY PLAN ' + pi_copy_flag_sql('events.rowid > ?'), (0,)))
        self.assertIn('SEARCH events USING INTEGER PRIMARY KEY (rowid>?)', plan)
        self.assertIn('SEARCH first_seen USING INDEX idx_events_pi_copy_key '
                      '(pi_copy_key=? AND rowid<?)', plan)

    def test_migration_flags_history_and_rebuilds_rollups_once(self):
        from app import db, main, summarizer
        base = self.baseline()
        self.post(self.pi('orig', 'proj-a/s1.jsonl', pi_events(self.t0)))
        self.post(self.pi('copy', 'proj-b/s1.jsonl', pi_events(self.t0, ids='x-')))
        ingested = self.conn.execute(
            'SELECT uuid, pi_copy_key, pi_copy FROM events ORDER BY uuid').fetchall()
        # Pre-release state: no keys, no flags, rollups that count both.
        self.conn.execute('UPDATE events SET pi_copy_key=NULL, pi_copy=NULL')
        self.conn.execute('UPDATE tool_uses SET pi_copy=NULL')
        self.conn.execute("DELETE FROM meta WHERE key='pi_copy_dedupe_version'")
        self.conn.commit()
        summarizer.summarize_days(sorted({r[0] for r in self.conn.execute(
            'SELECT DISTINCT day FROM events')}))
        self.assertAlmostEqual(self.conn.execute(
            'SELECT SUM(cost) FROM daily_summary').fetchone()[0], 2 * 1.85)
        db._migrate(self.conn)
        self.assertEqual(self.conn.execute(
            "SELECT value FROM meta WHERE key='daily_summary_rebuild_pending'").fetchone()[0], '1')
        # The migration derives the same keys and flags from stored rows.
        self.assertEqual([tuple(r) for r in ingested], [tuple(r) for r in self.conn.execute(
            'SELECT uuid, pi_copy_key, pi_copy FROM events ORDER BY uuid')])
        main._backfill_if_needed()
        self.assertEqual(self.metrics(), base)
        self.assertEqual(self.survivors(), ['orig'])
        # A second start neither re-flags nor re-queues a rebuild.
        db._migrate(self.conn)
        self.assertIsNone(self.conn.execute(
            "SELECT value FROM meta WHERE key='daily_summary_rebuild_pending'").fetchone())
        self.assertEqual(self.metrics(), base)


class PiCopyHAWindowTest(_HABase):
    def test_copied_pi_session_counts_once_in_window_spend(self):
        self._seed_oauth(fh_pct=50.0)
        t0 = self.now - 600
        with patch('app.aggregator.trigger_eager_rebuild'):
            http = self.client()
            self.addCleanup(http.close)
            for machine, ids in (('orig', ''), ('copy', 'x-')):
                body = {'machine': machine, 'account_class': 'personal',
                        'project_dir': '/' + machine, 'session_file': machine + '/s1.jsonl',
                        'events': pi_events(t0 - 130, ids=ids)}
                response = http.post('/api/ingest/pi', json=body,
                                     headers={'X-API-Key': self.api_key})
                self.assertEqual(response.status_code, 200, response.text)
            body = self._ha()
        # Claude spend only: $1.25 reply + $0.10 compaction, once.
        self.assertAlmostEqual(body['five_hour']['spend_usd'], 1.35, places=2)
        self.assertAlmostEqual(body['weekly']['spend_usd'], 1.35, places=2)
