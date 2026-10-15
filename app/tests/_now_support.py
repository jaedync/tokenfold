"""Event-row helpers shared by the /api/now and /api/today tests."""
import json

PERSONAL = {"account_email": "me@example.com", "plan": "max"}
WORK = {"account_email": "me@acme.io", "plan": "enterprise"}


def insert_event(conn, uuid, *, session, ts, req=None, type_="assistant",
                 model="claude-opus-4-8", machine="laptop", project="/w/proj",
                 inp=0, out=0, client="claude-code", provider=None,
                 account=PERSONAL, day=None, human=0, agent_id=None,
                 reported_total=None):
    """One event row. 1M input tokens of claude-opus-4-8 cost $5."""
    from datetime import datetime
    from app.config import TZ_NAME
    from zoneinfo import ZoneInfo
    day = day or datetime.fromtimestamp(ts, ZoneInfo(TZ_NAME)).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "message_id,source_machine,project_dir,model,is_sidechain,agent_id,"
        "input_tokens,output_tokens,account_email,plan,is_human_prompt,"
        "source_client,provider,reported_cost_total) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, type_, day + "T12:00:00Z", ts, day, session,
         req if req is not None else (uuid if type_ == "assistant" else None),
         None, machine, project, model, 0, agent_id, inp, out,
         account["account_email"], account["plan"], human, client, provider,
         reported_total))
    conn.commit()


def set_oauth(conn, usage: dict, observed: float, source="meridian-oauth"):
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
        (json.dumps({"data": usage, "updated_at": "", "source": source,
                     "observed_at_epoch": observed}),))
    conn.commit()
