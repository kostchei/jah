from jah.diagnostics.order_bias import _finding


def test_finding_treats_exact_ten_point_drop_as_wording_bias() -> None:
    runs = {
        "R0": {"cases": 50, "predicted_class_counts": {"other": 24}},
        "R1": {"other_prediction_share_by_position": {"A": 0.44, "B": 0.48, "C": 0.48, "D": 0.48}},
        "R3": {"cases": 50, "predicted_class_counts": {"other": 19}},
    }

    finding = _finding(runs)

    assert finding["branch"] == "wording-bias"
