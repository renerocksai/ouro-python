import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import boot


def complete(*output, tokens=20):
    return {"status": "completed", "output": list(output),
            "usage": {"output_tokens": tokens, "input_tokens": 10}}


def call(code, call_id="call_1"):
    return {"type": "function_call", "name": "eval", "call_id": call_id,
            "arguments": json.dumps({"code": code})}


LOOP = '''import json
def agent(prompt):
    history = [{"role": "user", "content": prompt}]
    while True:
        response = agent_prompt(history)
        history.extend(response["output"])
        calls = [x for x in response["output"] if x.get("type") == "function_call"]
        if not calls:
            return "".join(c["text"] for x in response["output"]
                           if x.get("type") == "message" for c in x["content"]
                           if c.get("type") == "output_text")
        for item in calls:
            result = agent_eval(json.loads(item["arguments"])["code"])
            history.append({"type": "function_call_output",
                            "call_id": item["call_id"], "output": result})
_result = agent("test task")
'''


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.payloads = []

    def harness(self, replies=None, **kwargs):
        replies = iter(replies or [])
        def transport(payload):
            self.payloads.append(copy.deepcopy(payload))
            return next(replies)
        return boot.Harness("secret-test-key", output_dir=self.directory.name,
                            transport=transport, **kwargs)

    def test_bootstrap_installs_actual_feedback_loop(self):
        reasoning = {"type": "reasoning", "id": "rs_1", "summary": [],
                     "encrypted_content": "opaque-reasoning"}
        final = {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "It works."}]}
        harness = self.harness([
            complete(call(LOOP)),
            complete(reasoning, call("persistent_value = 40\n_result = persistent_value + 2")),
            complete(call("persistent_value + 3", "call_2")),
            complete(final),
        ])
        seed = harness.scope["agent"]
        harness.boot("test task")
        self.assertIsNot(harness.scope["agent"], seed)
        self.assertEqual(harness.scope["persistent_value"], 40)
        self.assertEqual(harness.scope["_result"], "It works.")
        self.assertEqual(harness.requests, 4)
        history = self.payloads[2]["input"]
        self.assertIn(reasoning, history)
        self.assertEqual(history[-1], {"type": "function_call_output",
                                      "call_id": "call_1", "output": "42"})
        self.assertEqual(self.payloads[3]["input"][-1]["output"], "43")
        self.assertEqual(self.payloads[0]["model"], "gpt-6.1-sol")
        self.assertEqual(self.payloads[0]["reasoning"], {"effort": "medium"})
        self.assertFalse(self.payloads[0]["store"])
        self.assertEqual(self.payloads[0]["include"], ["reasoning.encrypted_content"])
        self.assertTrue((Path(self.directory.name) / "eval-0001.py").is_file())
        self.assertNotIn("opaque-reasoning", (Path(self.directory.name) / "usage.jsonl").read_text())

    def test_seed_does_not_request_tool_feedback(self):
        harness = self.harness([complete(call("seed_side_effect = True"))])
        harness.agent("one request only")
        self.assertEqual(harness.requests, 1)
        self.assertTrue(harness.scope["seed_side_effect"])

    def test_eval_error_is_feedback_and_globals_survive(self):
        harness = self.harness()
        terminal = io.StringIO()
        with patch("sys.stdout", terminal):
            self.assertEqual(harness.evaluate("x = 2\nprint('captured')\n_result = x"), "captured\n2")
        self.assertEqual(terminal.getvalue(), "captured\n")
        self.assertEqual(harness.evaluate("x + 1"), "3")
        self.assertIn("ZeroDivisionError", harness.evaluate("1 / 0"))
        self.assertEqual(harness.evaluate("pass"), "None")
        self.assertIn("SyntaxError", harness.evaluate("this is invalid Python!"))

    def test_request_limit_and_output_budget(self):
        harness = self.harness([complete(tokens=7), complete(tokens=3)],
                               max_output_tokens=8, total_output_tokens=10)
        harness.request([])
        harness.request([])
        self.assertEqual([p["max_output_tokens"] for p in self.payloads], [8, 3])
        with self.assertRaisesRegex(boot.OuroError, "budget exhausted"):
            harness.request([])
        other = self.harness([complete()], max_requests=1)
        other.request([])
        with self.assertRaisesRegex(boot.OuroError, "Request limit"):
            other.request([])

    def test_input_budget_blocks_before_transport(self):
        harness = self.harness(max_input_bytes=5)
        with self.assertRaisesRegex(boot.OuroError, "Input history"):
            harness.request([{"role": "user", "content": "hello"}])
        self.assertEqual(harness.requests, 0)

    def test_incomplete_response_is_never_evaluated(self):
        reply = complete(call("danger = True"), tokens=10)
        reply["status"] = "incomplete"
        reply["incomplete_details"] = {"reason": "max_output_tokens"}
        harness = self.harness([reply])
        with self.assertRaisesRegex(boot.OuroError, "did not complete"):
            harness.boot("test")
        self.assertNotIn("danger", harness.scope)
        self.assertEqual(harness.output_tokens, 10)

    def test_invalid_bootstrap_and_eval_failure_stop(self):
        cases = [complete(), complete(call("1/0")),
                 complete({"type": "function_call", "name": "eval", "arguments": "{}"}),
                 complete(call("pass"), call("pass"))]
        for response in cases:
            with self.subTest(response=response):
                harness = self.harness([response])
                with self.assertRaises(boot.OuroError):
                    harness.boot("test")

    def test_missing_usage_consumes_allocation(self):
        harness = self.harness([{"status": "completed", "output": []}],
                               max_output_tokens=5, total_output_tokens=5)
        harness.request([])
        self.assertEqual(harness.output_tokens, 5)
        with self.assertRaises(boot.OuroError):
            harness.request([])

    def test_failed_attempt_counts_and_is_not_retried(self):
        def fail(_):
            raise boot.OuroError("connection failed")
        harness = self.harness(max_requests=1)
        harness.transport = fail
        with self.assertRaisesRegex(boot.OuroError, "connection failed"):
            harness.request([])
        self.assertEqual(harness.requests, 1)
        with self.assertRaisesRegex(boot.OuroError, "Request limit"):
            harness.request([])

    def test_secret_redaction_and_source_permissions(self):
        harness = self.harness()
        self.assertEqual(harness.evaluate("'secret-test-key'"), "'[REDACTED]'")
        terminal = io.StringIO()
        with patch("sys.stdout", terminal):
            harness.evaluate("print('secret-test-key')")
        self.assertEqual(terminal.getvalue(), "[REDACTED]\n")
        source = Path(self.directory.name) / "eval-0001.py"
        self.assertNotIn("secret-test-key", source.read_text())
        self.assertEqual(source.stat().st_mode & 0o777, 0o600)

    def test_dotenv_read_does_not_execute_shell(self):
        path = Path(self.directory.name) / ".env"
        path.write_text("OTHER=ignored\nexport OPENAI_API_KEY='literal-$(no-shell)'\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(boot.read_api_key(path), "literal-$(no-shell)")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "from-environment"}, clear=True):
            self.assertEqual(boot.read_api_key(path), "from-environment")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(boot.OuroError):
                boot.read_api_key(path.parent / "missing")


if __name__ == "__main__":
    unittest.main()
