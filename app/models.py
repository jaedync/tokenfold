from pydantic import BaseModel


class CursorState(BaseModel):
    last_line_num: int = 0


class IngestRequest(BaseModel):
    machine: str
    project_dir: str
    session_file: str
    cursor: CursorState = CursorState()
    events: list[dict]


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int
    cursor: CursorState
