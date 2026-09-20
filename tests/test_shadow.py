
from jah.schemas import BooleanAnswer, EvaluateRequest, EvaluateResponse, Timing, Usage
from jah.shadow import ShadowEvaluator, analyze_shadow_traffic


def make_mock_response(req_id: str, value: bool, disposition: str = "accept") -> EvaluateResponse:
    return EvaluateResponse(
        request_id=req_id,
        artifact_id="test-art",
        profile_id="test-profile",
        answers={
            "check": BooleanAnswer(
                type="boolean",
                value=value,
                p_true=0.95 if value else 0.05,
                probabilities={"false": 0.05 if value else 0.95, "true": 0.95 if value else 0.05},
                calibration_status="validated",
                disposition=disposition,
            )
        },
        usage=Usage(unique_state_tokens=10, processed_input_tokens=20, decisions=1),
        timing_ms=Timing(queue=1.0, inference=10.0, total=11.0),
    )


def test_shadow_evaluator_records_and_analyzes(tmp_path):
    log_path = tmp_path / "shadow.jsonl"
    evaluator = ShadowEvaluator(log_path)

    req1 = EvaluateRequest.model_validate({
        "state": "State 1",
        "questions": {"check": {"type": "boolean", "instructions": "Check", "proposition": "P1"}},
    })
    resp1 = make_mock_response("req-1", value=True, disposition="accept")
    evaluator.record(req1, resp1)

    req2 = EvaluateRequest.model_validate({
        "state": "State 2",
        "questions": {"check": {"type": "boolean", "instructions": "Check", "proposition": "P2"}},
    })
    resp2 = make_mock_response("req-2", value=False, disposition="review")
    evaluator.record(req2, resp2)

    # Analyze without ground truth
    stats = analyze_shadow_traffic(log_path)
    assert stats["total_requests"] == 2
    assert stats["total_decisions"] == 2
    assert stats["coverage"] == 0.5
    assert stats["review_rate"] == 0.5
    assert stats["disposition_counts"] == {"accept": 1, "review": 1}
    assert stats["evaluated_against_ground_truth"] is False

    # Analyze with ground truth: req-1 matches True (agreement 1/1 on labeled), req-2 ground truth True vs predicted False
    ground_truth = {
        "req-1": {"check": "true"},
        "req-2": {"check": "true"},  # ground truth disagreed with prediction False
    }
    stats_gt = analyze_shadow_traffic(log_path, ground_truth_labels=ground_truth)
    assert stats_gt["evaluated_against_ground_truth"] is True
    assert stats_gt["ground_truth_samples"] == 2
    assert stats_gt["agreement_rate"] == 0.5
