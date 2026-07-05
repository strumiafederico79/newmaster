import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_assistant import build_fallback_response


def test_brillo_suggestion_is_actionable():
    analysis = {
        "lufs": -14.2,
        "spectrum": {"air": -28.0, "presence": -19.0},
        "mix_advice": {"issues": ["Altas frecuencias muy bajas"], "tips": ["Subí el high shelf"]},
    }

    result = build_fallback_response("quiero más brillo", analysis)

    assert result["suggested_params"]["high_shelf_gain_db"] > 0
    assert result["suggested_params"]["eq4_gain"] > 0
    assert result["suggestion_summary"] is not None


def test_loudness_request_uses_target_lufs_or_makeup_gain():
    analysis = {"lufs": -20.5, "peak_db": -1.0, "true_peak_db": -0.2}

    result = build_fallback_response("hazlo más fuerte", analysis)

    params = result["suggested_params"]
    assert params
    assert "target_lufs" in params or "comp_makeup_db" in params or "limiter_ceiling" in params


def test_eq_instruction_is_parsed_into_parameter_changes():
    analysis = {}

    result = build_fallback_response("dame 2 db menos en 4k q 1.2", analysis)

    params = result["suggested_params"]
    assert params["eq4_gain"] == -2.0
    assert params["eq4_freq"] == 4000.0
    assert params["eq4_q"] == 1.2


def test_advanced_dsp_mode_is_exposed_to_the_assistant():
    analysis = {}

    result = build_fallback_response("usa linear phase para la eq", analysis)

    params = result["suggested_params"]
    assert params["eq_mode"] == "linear_phase"


def test_boolean_dsp_blocks_are_exposed_to_the_assistant():
    analysis = {}

    result = build_fallback_response("desactiva el glue compressor", analysis)

    params = result["suggested_params"]
    assert params["glue_bypass"] is True
