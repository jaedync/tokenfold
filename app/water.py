"""Water & energy footprint estimation for LLM inference.

Methodology and sources:
- Energy per token: Epoch AI (2024), "How Hungry is AI?" (arXiv:2505.09598),
  Simon Couch's Claude Code analysis (2026)
- PUE: AWS Sustainability Report 2024 (1.15), Google Environmental Report 2024 (1.09)
- WUE: AWS (0.15 L/kWh), Google (~1.04 L/kWh facility avg)
- Grid water intensity: EIA (2024), adjusted for hyperscaler renewable procurement

Anthropic infrastructure (confirmed public sources):
- AWS: Trainium2 (Project Rainier - Indiana/Pennsylvania/Mississippi)
  Cooling: direct-to-chip liquid cooling, closed-loop — no additional water consumption
- Google Cloud: TPU v6 Trillium, scaling to Ironwood (TPU v7)
  Cooling: liquid cooling with split-flow cold plates since TPU v3, per-rack PUE 1.02-1.03
- Azure: NVIDIA Grace Blackwell / Vera Rubin GPUs
  Cooling: transitioning to liquid cooling for AI racks, less transparency on WUE
"""

# Energy estimates in Wh per million tokens.
# Output (autoregressive decode) vs input (parallel prefill).
#
# Calibrated from:
# - "How Hungry is AI?" (2025): Claude 3.7 Sonnet measured at 5.518 Wh for
#   10K input + 1.5K output tokens -> ~2,000 Wh/MTok output after isolating
#   decode energy (output tokens dominate: ~5-11x input per "Prompts to Power")
# - Epoch AI: GPT-4o ~0.3 Wh for 500 output tokens at ~100B active params
# - Simon Couch (2026): Claude Code estimates of 1,950 Wh/MTok output,
#   390 Wh/MTok input for Sonnet-class models
# - Cross-model scaling via pricing ratios as compute proxy:
#   Opus ($25/MTok) : Sonnet ($15/MTok) : Haiku ($5/MTok) = 5:3:1
#   Adjusted for larger models being less efficient per-token (sublinear scaling)
MODEL_ENERGY_WH_PER_MTOK = {
    # (output_wh_per_mtok, input_wh_per_mtok)
    "Opus 4.6":   (2_800, 560),
    "Opus 4.5":   (2_800, 560),
    "Sonnet 4.6": (1_950, 390),
    "Sonnet 4.5": (1_950, 390),
    "Sonnet 4":   (1_950, 390),
    "Sonnet 3.5": (1_950, 390),
    "Haiku 4.5":  (650, 130),
    "Haiku 3.5":  (520, 104),
}
_FALLBACK_ENERGY = (1_950, 390)  # Sonnet-class default

# Data center efficiency factors.
#
# Anthropic uses a mix of AWS, Google Cloud, and Azure.
# PUE (Power Usage Effectiveness) - ratio of total facility power to IT power:
#   AWS global avg: 1.15 (2024 Sustainability Report)
#   Google global avg: 1.09 (2024 Environmental Report)
#   Blended estimate: 1.12
PUE = 1.12

# WUE (Water Usage Effectiveness) — on-site evaporative cooling water.
#
# IMPORTANT: Facility-level WUE averages overstate water use for AI inference
# because modern AI accelerator racks use liquid cooling, not evaporative:
#
#   AWS Trainium2 (Project Rainier):
#     Direct-to-chip liquid cooling in a CLOSED-LOOP system.
#     AWS states this "does not increase water consumption."
#     Effective rack-level WUE: ~0 L/kWh for the cooling loop itself.
#     Facility still uses some water for ambient/common cooling.
#     AWS overall WUE: 0.15 L/kWh (already very low, reflects liquid cooling).
#     Estimate for AI racks: ~0.08 L/kWh
#
#   Google TPU v6+ (Trillium/Ironwood):
#     Liquid cooling with split-flow cold plates since TPU v3 (2018).
#     Removes 70-75% of rack heat through liquid (not evaporative).
#     Per-rack PUE: 1.02-1.03. Facility avg WUE: ~1.04 L/kWh.
#     But TPU liquid-cooled racks use far less water than facility avg.
#     Estimate for TPU racks: ~0.30 L/kWh
#
#   Azure NVIDIA GPUs (Grace Blackwell / Vera Rubin):
#     Microsoft historically has higher WUE (~1.8 L/kWh in some reports).
#     Transitioning to liquid cooling for AI clusters but less transparent.
#     Estimate for AI racks: ~0.70 L/kWh
#
# Blended estimate for AI inference racks (not facility averages):
#   (0.08 + 0.30 + 0.70) / 3 ≈ 0.36 L/kWh
WUE_ONSITE_L_PER_KWH = 0.36

# Off-site water intensity of electricity generation:
#   US grid average: ~1.8 L/kWh (EIA 2024)
#   Adjusted for hyperscaler renewable procurement (~60%+ wind/solar = near-zero
#   water) and natural gas remainder (~0.7 L/kWh vs coal at ~5.4 L/kWh)
WATER_GRID_L_PER_KWH = 0.50

# Combined water factor: mL of water per Wh of compute energy
# = PUE * (WUE_onsite + grid_water) in L/kWh = same number in mL/Wh
WATER_ML_PER_WH = PUE * (WUE_ONSITE_L_PER_KWH + WATER_GRID_L_PER_KWH)


def _get_energy(model_name: str) -> tuple[float, float]:
    """Return (output_wh_per_mtok, input_wh_per_mtok) for a model."""
    if model_name in MODEL_ENERGY_WH_PER_MTOK:
        return MODEL_ENERGY_WH_PER_MTOK[model_name]
    n = model_name.lower()
    if "opus" in n:
        return MODEL_ENERGY_WH_PER_MTOK["Opus 4.6"]
    if "haiku" in n:
        return MODEL_ENERGY_WH_PER_MTOK["Haiku 4.5"]
    return _FALLBACK_ENERGY


def compute_energy_wh(model_name: str, input_tok: int, output_tok: int) -> float:
    """Estimate energy in Wh for a given token count."""
    out_rate, in_rate = _get_energy(model_name)
    return (output_tok * out_rate + input_tok * in_rate) / 1_000_000


def compute_water_ml(model_name: str, input_tok: int, output_tok: int) -> float:
    """Estimate water consumption in mL for a given token count."""
    return compute_energy_wh(model_name, input_tok, output_tok) * WATER_ML_PER_WH
