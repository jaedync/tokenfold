from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    cache_tiers: uuid -> [ephemeral_5m, ephemeral_1h];
    server_tools: uuid -> [web_search_requests, web_fetch_requests];
    sig_headers: uuid -> [sig_version, sig_header_b64, sig_cipher_len];
    titles: session_id -> title.
    Batches are capped — the client splits large backfills across requests."""
    cache_tiers: dict[str, list[int]] = Field(default_factory=dict, max_length=20000)
    server_tools: dict[str, list[int]] = Field(default_factory=dict, max_length=20000)
    # Untyped list: the triple is mixed (int, str, int). The server validates
    # each element itself; see _sig_columns in app/ingest.py.
    sig_headers: dict[str, list] = Field(default_factory=dict, max_length=20000)
    titles: dict[str, str] = Field(default_factory=dict, max_length=20000)
    # Multi-batch protocol: data batches send reroll=False (server defers the
    # expensive day re-roll and just reports touched days); the client then
    # sends ONE final request with the union as reroll_days, so each affected
    # day is summarized exactly once instead of once per batch.
    reroll: bool = True
    reroll_days: list[str] = Field(default_factory=list, max_length=2000)


class EnterpriseBudgetRequest(BaseModel):
    """Enterprise monthly $ budget setting. null clears the stored value."""
    budget_usd: float | None = None


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


class PiCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_line_num: int = Field(default=0, ge=0, le=10**12)


class PiUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: int = Field(default=0, ge=0, le=10**12)
    output: int = Field(default=0, ge=0, le=10**12)
    cache_read: int = Field(default=0, ge=0, le=10**12)
    cache_write: int = Field(default=0, ge=0, le=10**12)
    reasoning: int = Field(default=0, ge=0, le=10**12)
    cost_input: float | None = Field(default=None, ge=0, le=10**9)
    cost_output: float | None = Field(default=None, ge=0, le=10**9)
    cost_cache_read: float | None = Field(default=None, ge=0, le=10**9)
    cost_cache_write: float | None = Field(default=None, ge=0, le=10**9)
    cost_total: float | None = Field(default=None, ge=0, le=10**9)


class PiTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_use_id: str = Field(min_length=1, max_length=512)
    name: str = Field(min_length=1, max_length=512)


class PiEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=512)
    timestamp: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=512)
    parent_event_id: str | None = Field(default=None, max_length=512)
    kind: Literal["user", "assistant", "tool_result", "compaction", "branch_summary", "tool_usage"]
    provider: str | None = Field(default=None, max_length=128)
    api: str | None = Field(default=None, max_length=256)
    model: str | None = Field(default=None, max_length=256)
    request_id: str | None = Field(default=None, max_length=512)
    agent_id: str | None = Field(default=None, max_length=512)
    is_sidechain: bool = False
    stop_reason: str | None = Field(default=None, max_length=128)
    usage: PiUsage | None = None
    has_text: bool = False
    has_thinking: bool = False
    has_tool_use: bool = False
    has_tool_result: bool = False
    has_image: bool = False
    text_length: int = Field(default=0, ge=0, le=10**9)
    thinking_length: int = Field(default=0, ge=0, le=10**9)
    tools: list[PiTool] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def require_usage_identity(self):
        """Usage must retain provider/model identity for safe pricing."""
        if self.usage is not None and (not self.provider or not self.model):
            raise ValueError("usage-bearing events require provider and model")
        return self


class PiIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine: str = Field(min_length=1, max_length=128)
    account_class: Literal["work", "personal"]
    project_dir: str = Field(min_length=1, max_length=1024)
    session_file: str = Field(min_length=1, max_length=1024)
    cursor: PiCursor = Field(default_factory=PiCursor)
    events: list[PiEvent] = Field(max_length=5000)


class ProviderLimitWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")
    label: str = Field(min_length=1, max_length=64)
    pct: float = Field(ge=0, le=100)
    resets_at_epoch: float | None = Field(default=None, ge=0, le=10**11)
    window_seconds: int | None = Field(default=None, ge=1, le=10 * 365 * 86400)


class ProviderLimitSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["codex", "opencode-go", "opencode-zen"]
    observed_at_epoch: float | None = Field(default=None, ge=0, le=10**11)
    windows: list[ProviderLimitWindow] = Field(default_factory=list, max_length=8)
    # The provider's own plan label ("plus", "enterprise", ...), shown as-is
    # and useful for spotting a misclassified machine at a glance.
    plan: str | None = Field(default=None, max_length=32, pattern=r"^[a-z0-9_-]+$")


class ProviderUsageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine: str = Field(min_length=1, max_length=128)
    # Same fleet classification Pi event batches carry; it selects the
    # dashboard scope the snapshots land in. Required so pre-scope clients
    # fail closed instead of stomping a peer scope.
    account_class: Literal["work", "personal"]
    limits: list[ProviderLimitSnapshot] = Field(max_length=3)


class ClaudeUsageBucket(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    key: str = Field(max_length=64, pattern=r"^(five_hour|seven_day|scoped:[a-z0-9]+(?:_[a-z0-9]+)*)$")
    label: str = Field(min_length=1, max_length=64, pattern=r"\S")
    pct: float = Field(ge=0, le=100)
    resets_at_epoch: float = Field(gt=0, le=10**11)


class ClaudeExtraUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    enabled: bool
    monthly_limit_cents: float | None = Field(default=None, ge=0, le=10**9)
    used_cents: float | None = Field(default=None, ge=0, le=10**9)
    pct: float | None = Field(default=None, ge=0, le=100)


class ClaudeUsageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    machine: str = Field(min_length=1, max_length=128, pattern=r"\S")
    account_class: Literal["personal"]
    source: Literal["meridian-oauth"]
    source_profile: Literal["default"]
    observed_at_epoch: float = Field(gt=0, le=10**11)
    buckets: list[ClaudeUsageBucket] = Field(min_length=2, max_length=16)
    extra_usage: ClaudeExtraUsage | None = None

    @model_validator(mode="after")
    def unique_required_buckets(self):
        keys = [b.key for b in self.buckets]
        if len(set(keys)) != len(keys) or not {"five_hour", "seven_day"} <= set(keys):
            raise ValueError("unique five_hour and seven_day buckets required")
        return self
