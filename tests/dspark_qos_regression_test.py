#!/usr/bin/env python3
"""Host unit tests for the DSpark QoS harness's evidence checks (no server/GPU)."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "scripts" / "dspark_qos_regression.py"
spec = importlib.util.spec_from_file_location("dspark_qos", path)
qos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qos)


class HarnessChecks(unittest.TestCase):
    def test_counter_delta_requires_both_samples_and_rejects_reset(self):
        self.assertEqual(qos.metric_delta({"x": 4}, {"x": 7}, "x"), 3)
        for before, after in (({}, {"x": 7}), ({"x": 4}, {}), ({"x": 7}, {"x": 4})):
            with self.assertRaises(qos.HarnessError):
                qos.metric_delta(before, after, "x")

    def test_calibration_brackets_safely_then_finds_largest_prompt(self):
        seen = []
        phrase = "rollover-boundary marker "
        def fake(_base, _model, messages, **_options):
            count = len(messages[0]["content"]) // len(phrase)
            seen.append(count)
            if count > 64:
                raise AssertionError("calibration overflowed a short server context")
            return {"prompt_tokens": 10 + count * 3}
        with patch.object(qos, "chat_request", side_effect=fake):
            prompt, tokens = qos.calibrated_rollover_prompt("", "", None, 1, target=100)
        self.assertEqual(len(prompt) // len(phrase), 30)
        self.assertEqual(tokens, 100)
        self.assertEqual(seen[:5], [1, 2, 4, 8, 16])
        self.assertLessEqual(max(seen), 32)

    def test_missing_usage_refuses_calibration(self):
        with patch.object(qos, "chat_request", return_value={}):
            with self.assertRaises(qos.HarnessError):
                qos.calibrated_rollover_prompt("", "", None, 1)

    def test_stream_first_event_is_not_first_text(self):
        event_only = b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: [DONE]\n\n'
        with patch.object(qos, "fetch", return_value=io.BytesIO(event_only)):
            r = qos.stream_request("", "", "test", api_key=None, timeout=1)
        self.assertEqual(r["events"], 1)
        self.assertIsNotNone(r["first_event_s"])
        self.assertIsNone(r["first_text_s"])
        reasoning = b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\ndata: [DONE]\n\n'
        with patch.object(qos, "fetch", return_value=io.BytesIO(reasoning)):
            r = qos.stream_request("", "", "test", api_key=None, timeout=1)
        self.assertIsNotNone(r["first_text_s"])

    def test_draft_token_total_reports_observed_trials(self):
        self.assertEqual(qos.observed_draft_token_total([{"draft_tokens": 168}, {"draft_tokens": 189}]), 357)
        self.assertEqual(qos.observed_draft_token_total([]), 0)

    def test_seeded_repeat_requires_nonempty_matching_content_and_usage(self):
        sample = {"content": "A bright comet crosses the sky.", "finish_reason": "stop",
                  "completion_tokens": 7}
        self.assertTrue(qos.seeded_sample_reproduced(sample, dict(sample)))
        for first, repeat in (
            ({**sample, "content": ""}, {**sample, "content": ""}),
            ({**sample, "content": "   "}, {**sample, "content": "   "}),
            (sample, {**sample, "content": "A different comet."}),
            (sample, {**sample, "finish_reason": "length"}),
            (sample, {**sample, "completion_tokens": 8}),
            (sample, {**sample, "completion_tokens": 7.0}),
            (sample, {**sample, "completion_tokens": None}),
            ({**sample, "completion_tokens": 1}, {**sample, "completion_tokens": True}),
            ({**sample, "finish_reason": None}, {**sample, "finish_reason": None}),
        ):
            with self.subTest(first=first, repeat=repeat):
                self.assertFalse(qos.seeded_sample_reproduced(first, repeat))

    def test_chat_request_forwards_reasoning_effort(self):
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                   "usage": {"completion_tokens": 1}}
        with patch.object(qos, "fetch", return_value=io.BytesIO(json.dumps(payload).encode())) as fetch:
            result = qos.chat_request("http://localhost", "spark", [{"role": "user", "content": "test"}],
                                      api_key=None, timeout=1, reasoning_effort="none")
        self.assertEqual(result["content"], "OK")
        sent = json.loads(fetch.call_args.kwargs["data"])
        self.assertEqual(sent["reasoning_effort"], "none")

    def test_stop_requires_control_and_exact_prefix(self):
        marker = "STOP"
        stopped = {"finish_reason": "stop", "content": "prefix "}
        self.assertTrue(qos.stop_sequence_observed({"content": "prefix STOP after"}, stopped, marker))
        self.assertFalse(qos.stop_sequence_observed({"content": "prefix "}, stopped, marker))
        self.assertFalse(qos.stop_sequence_observed({"content": "prefix STOP"},
                                                     {"finish_reason": "length", "content": "prefix "}, marker))
        self.assertFalse(qos.stop_sequence_observed({"content": "prefix STOP"},
                                                     {"finish_reason": "stop", "content": "other"}, marker))


if __name__ == "__main__":
    unittest.main()
