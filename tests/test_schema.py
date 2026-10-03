from hazop_agent.criteria import coverage_key, load_criteria, parse_deviation, suggested_sil
from hazop_agent.schema import FIELDS, validate_row

CRIT = load_criteria()


def good(**kw):
    row = {"node": "N", "deviation": "Less Flow | feed", "causes": "Valve fails closed", "consequences": "Loss of feed",
           "safeguards": ["Low flow alarm"], "FI": 4, "SI": 3, "RI": 12, "SIL": suggested_sil(CRIT, 4, 3),
           "recommendations": []}
    row.update(kw)
    return row


def test_valid_row_passes_unchanged():
    row, errors, warnings = validate_row(good(), CRIT)
    assert errors == [] and warnings == []
    assert list(row) == FIELDS and row["RI"] == 12


def test_ri_must_equal_fi_times_si():
    row, errors, _ = validate_row(good(RI=10), CRIT)
    assert row is None and any("RI" in e for e in errors)


def test_ranges_and_types():
    for bad in ({"FI": 0}, {"SI": 6}, {"FI": "high"}, {"FI": True}, {"SIL": 7}, {"causes": ""}, {"node": None},
                {"safeguards": 5}):
        row, errors, _ = validate_row(good(**bad), CRIT)
        assert row is None and errors, bad


def test_missing_ri_and_sil_are_filled_with_a_warning():
    raw = good()
    del raw["RI"], raw["SIL"]
    row, errors, warnings = validate_row(raw, CRIT)
    assert not errors and row["RI"] == 12 and row["SIL"] == suggested_sil(CRIT, 4, 3) and len(warnings) == 2


def test_sil_off_matrix_is_accepted_with_warning():
    row, errors, warnings = validate_row(good(SIL=3), CRIT)
    assert row["SIL"] == 3 and not errors and any("differs" in w for w in warnings)


def test_string_lists_and_numeric_strings_are_normalised():
    row, errors, warnings = validate_row(good(safeguards="PSV-1; alarm\n- interlock", FI="4", extra=1), CRIT)
    assert not errors and row["safeguards"] == ["PSV-1", "alarm", "interlock"] and row["FI"] == 4
    assert any("unknown" in w for w in warnings)


def test_not_a_dict():
    assert validate_row(["x"], CRIT)[0] is None


def test_parse_deviation_and_coverage_keys():
    assert parse_deviation("Low Temperature | stack") == ("Less", "Temperature")
    assert parse_deviation("HIGH pressure | outlet") == ("More", "Pressure")
    assert parse_deviation("Low power output") == ("Less", "Power")
    assert parse_deviation("Reverse flow | vent") == ("Reverse", "Flow")
    assert parse_deviation("As well as composition") == ("As well as", "Composition")
    assert coverage_key("No Flow | air") == "Less Flow"
    assert coverage_key("Something odd") is None


def test_sil_matrix_shape():
    for si in range(1, 6):
        for fi in range(1, 6):
            assert 0 <= suggested_sil(CRIT, fi, si) <= 4
