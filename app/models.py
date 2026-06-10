from pydantic import BaseModel, Field


class CursorState(BaseModel):
    last_line_num: int = 0


class IngestRequest(BaseModel):
    machine: str = Field(max_length=128)
    project_dir: str = Field(max_length=1024)
    session_file: str = Field(max_length=1024)
    cursor: CursorState = CursorState()
    events: list[dict]
    account_email: str | None = Field(None, max_length=320)
    org_name: str | None = Field(None, max_length=256)
    plan: str | None = Field(None, max_length=64)
    rate_limit_tier: str | None = Field(None, max_length=64)
    org_type: str | None = Field(None, max_length=64)
    org_uuid: str | None = Field(None, max_length=64)


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int
    cursor: CursorState


class BackfillRequest(BaseModel):
    """Historical repair payload generated from a machine's local transcripts.
    cache_tiers: uuid -> [ephemeral_5m, ephemeral_1h]; titles: session_id -> title.
    Batches are capped — the client splits large backfills across requests."""
    cache_tiers: dict[str, list[int]] = Field(default_factory=dict, max_length=20000)
    titles: dict[str, str] = Field(default_factory=dict, max_length=20000)
    # Multi-batch protocol: data batches send reroll=False (server defers the
    # expensive day re-roll and just reports touched days); the client then
    # sends ONE final request with the union as reroll_days, so each affected
    # day is summarized exactly once instead of once per batch.
    reroll: bool = True
    reroll_days: list[str] = Field(default_factory=list, max_length=2000)


class DesktopSessionUpsert(BaseModel):
    cli_session_id: str
    desktop_session_id: str | None = None
    title: str | None = None
    model: str | None = None
    effort: str | None = None
    permission_mode: str | None = None
    completed_turns: int | None = None
    is_archived: bool | None = None
    cwd: str | None = None
    origin_cwd: str | None = None
    created_at_ms: int | None = None
    last_activity_at_ms: int | None = None
    enabled_mcp_tools: dict | None = None
    remote_mcp_servers: list | None = None
    chrome_permission_mode: str | None = None
    chrome_allowed_domains: list[str] | None = None


class DesktopMetadataRequest(BaseModel):
    machine: str
    sessions: list[DesktopSessionUpsert]


class DesktopMetadataResponse(BaseModel):
    inserted: int
    updated: int
    ignored_stale: int
