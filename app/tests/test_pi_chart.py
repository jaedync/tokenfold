"""Frontend contract for provider-reported Pi model costs."""
from pathlib import Path


TEMPLATE = (Path(__file__).resolve().parents[2] / "templates" / "dashboard.html")


def test_cost_chart_keeps_reported_models_and_components():
    source = TEMPLATE.read_text()
    assert "m[activeCostKey] > 0" in source
    assert "recent_cost_cache_write_reported" in source
    assert "recent_cost_other" in source
    assert "label:'Reported Cache Write'" in source
    assert "label:'Other'" in source
