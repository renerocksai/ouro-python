#!/usr/bin/env python3
"""One request, one eval: the model writes the agent loop in stock Python."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


class OuroError(RuntimeError):
    pass


class _Tee(io.StringIO):
    """Return printed tool output while keeping generated terminal UIs visible."""

    def __init__(self, stream, redact):
        super().__init__()
        self.stream, self.redact = stream, redact

    def write(self, text):
        self.stream.write(self.redact(text))
        return super().write(text)

    def flush(self):
        self.stream.flush()


def read_api_key(env_file):
    """Read only OPENAI_API_KEY, without sourcing or executing the dotenv file."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    path = Path(env_file)
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].rstrip()
                if value:
                    return value
    raise OuroError("Set OPENAI_API_KEY or put it in the selected .env file.")


EVAL_TOOL = {
    "type": "function",
    "name": "eval",
    "description": "Evaluate Python code in the running interpreter's shared globals. "
                   "An expression returns repr(value); statements may set _result. "
                   "Printed output and exceptions are returned as text.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": True,
}


class Harness:
    """Transport, eval, and usage limits; deliberately no agent feedback loop."""

    def __init__(self, api_key, *, model="gpt-6.1-sol", reasoning="medium",
                 max_requests=8, max_output_tokens=4096,
                 total_output_tokens=16384, max_input_bytes=120000,
                 timeout=120, output_dir=None, transport=None):
        for name, value in (("max_requests", max_requests),
                            ("max_output_tokens", max_output_tokens),
                            ("total_output_tokens", total_output_tokens),
                            ("max_input_bytes", max_input_bytes), ("timeout", timeout)):
            if value <= 0:
                raise OuroError(f"{name} must be positive.")
        self._api_key = api_key
        self.model, self.reasoning = model, reasoning
        self.max_requests, self.max_output_tokens = max_requests, max_output_tokens
        self.total_output_tokens, self.max_input_bytes = total_output_tokens, max_input_bytes
        self.timeout, self.transport = timeout, transport or self._post
        self.requests = self.output_tokens = self.evals = 0
        self.output_dir = Path(output_dir or Path.cwd() / ".ouro-python" /
                               f"run-{time.time_ns()}")
        self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.scope = {
            "__name__": "ouro_runtime", "__builtins__": __builtins__,
            "agent_prompt": self.request, "agent_eval": self.evaluate,
            "agent": self.agent, "eval_tool": EVAL_TOOL,
        }

    def redact(self, text):
        return str(text).replace(self._api_key, "[REDACTED]") if self._api_key else str(text)

    def _post(self, payload):
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": "Bearer " + self._api_key,
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read()).get("error", {}).get("message", "")
            except (ValueError, AttributeError):
                message = "API rejected the request."
            raise OuroError(self.redact(f"HTTP {error.code}: {message}")) from None
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise OuroError(self.redact(f"API request failed: {error}")) from None

    def request(self, messages):
        """Return the full Responses object; callers must replay ALL output items."""
        if self.requests >= self.max_requests:
            raise OuroError(f"Request limit reached ({self.max_requests}).")
        remaining = self.total_output_tokens - self.output_tokens
        if remaining <= 0:
            raise OuroError("Total output token budget exhausted.")
        if len(json.dumps(messages).encode()) > self.max_input_bytes:
            raise OuroError("Input history exceeds max_input_bytes.")
        requested = min(self.max_output_tokens, remaining)
        payload = {
            "model": self.model, "reasoning": {"effort": self.reasoning},
            "store": False, "include": ["reasoning.encrypted_content"],
            "tools": [EVAL_TOOL], "parallel_tool_calls": False,
            "max_output_tokens": requested, "input": messages,
        }
        self.requests += 1  # Failed attempts count too; there are no hidden retries.
        response = self.transport(payload)
        usage = response.get("usage") or {}
        # Missing usage conservatively consumes the entire requested allocation.
        used = usage.get("output_tokens", requested)
        if not isinstance(used, int) or used < 0:
            raise OuroError("API returned invalid token usage.")
        self.output_tokens += used
        trace = {"request": self.requests, "model": self.model,
                 "usage": usage, "total_output_tokens": self.output_tokens,
                 "status": response.get("status")}
        self._write("usage.jsonl", json.dumps(trace) + "\n", append=True)
        if response.get("status") != "completed":
            raise OuroError(self.redact("Response did not complete: " +
                                       str(response.get("incomplete_details") or
                                           response.get("error") or response.get("status"))))
        if not isinstance(response.get("output"), list):
            raise OuroError("API response has no output array.")
        return response

    def _write(self, name, text, *, append=False):
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
        with os.fdopen(os.open(self.output_dir / name, flags, 0o600), "w") as stream:
            stream.write(self.redact(text))

    def evaluate(self, code, *, capture=True, propagate=False):
        """Evaluate in one persistent namespace and save source before execution."""
        self.evals += 1
        name = f"eval-{self.evals:04d}.py"
        self._write(name, code)
        output = _Tee(sys.stdout, self.redact) if capture else io.StringIO()
        self.scope.pop("_result", None)
        try:
            try:
                compiled = compile(code, str(self.output_dir / name), "eval")
            except SyntaxError:
                compiled = compile(code, str(self.output_dir / name), "exec")
                expression = False
            else:
                expression = True
            with contextlib.redirect_stdout(output) if capture else contextlib.nullcontext():
                if expression:
                    result = eval(compiled, self.scope, self.scope)
                else:
                    exec(compiled, self.scope, self.scope)
                    result = self.scope.get("_result")
            value = output.getvalue() + repr(result)
            return self.redact(value)
        except Exception as error:
            message = self.redact(f"{type(error).__name__}: {error}")
            if propagate:
                raise OuroError(message) from None
            return self.redact(output.getvalue()) + "Error: " + message

    def agent(self, prompt):
        """Seed: make one request and execute one eval, dropping its tool result."""
        response = self.request([{"role": "user", "content": prompt}])
        calls = [item for item in response["output"] if item.get("type") == "function_call"]
        if len(calls) != 1 or calls[0].get("name") != "eval":
            raise OuroError("Bootstrap must return exactly one eval call.")
        try:
            arguments = json.loads(calls[0]["arguments"])
            code = arguments["code"]
            if not isinstance(code, str) or set(arguments) != {"code"}:
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise OuroError("Bootstrap returned invalid eval arguments.") from None
        # Bootstrap calls the new loop, whose terminal UI must remain visible.
        self.evaluate(code, capture=False, propagate=True)

    def boot(self, task):
        source = Path(__file__).read_text()
        seed = self.scope["agent"]
        prompt = f"""You run inside stock Python. Your only tool is eval, which executes
Python in shared globals. This is your entire harness (no existing agent loop):

{source}

This is a bootstrap. The current agent makes ONE request, executes ONE eval for
side effects, and drops its result. You cannot observe an eval result on this
call, so do not explore. In a single eval redefine the GLOBAL function agent:
make requests via agent_prompt(history), append EVERY response['output'] item
unchanged (including reasoning/encrypted_content), execute eval function calls
via agent_eval(json.loads(call['arguments'])['code']), append each result as
{{'type':'function_call_output','call_id':call['call_id'],'output':result}},
then loop until the model replies without a function call. Import json if needed.
Return/print assistant output_text on a final response. Catch eval errors through
agent_eval; let API/budget failures stop. Keep code short and use only stdlib.
The runtime globals persist between evals. This is real local Python, not a sandbox.
Then CALL YOUR NEW agent with the user's task, and print its return if not None:

{task}
"""
        seed(prompt)
        if self.scope["agent"] is seed:
            raise OuroError("Bootstrap did not replace the global agent function.")


DEFAULT_TASK = """Build and start a simple terminal chat in this interpreter.
Use input('You: ') and print; accept /quit or EOF to exit. For each user turn,
use the agent loop you created, sharing conversational history across turns.
Tell the assistant it may use eval to inspect and change this Python environment.
Do not read files or make network calls other than agent_prompt unless requested.
Keep it short. The chat must run now, not just be described."""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning", choices=("medium", "high"), default="medium")
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--total-output-tokens", type=int, default=16384)
    parser.add_argument("--max-input-bytes", type=int, default=120000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    harness = None
    try:
        options = vars(args).copy()
        task, env_file = options.pop("task"), options.pop("env_file")
        harness = Harness(read_api_key(env_file), **options)
        print(f"Generated code and usage: {harness.output_dir}", file=sys.stderr)
        harness.boot(task)
        return 0
    except (OuroError, KeyboardInterrupt, EOFError) as error:
        message = "Interrupted." if isinstance(error, KeyboardInterrupt) else str(error)
        print("ouro-python: " + message, file=sys.stderr)
        return 1
    finally:
        if harness:
            print(f"Requests: {harness.requests}; output tokens: {harness.output_tokens}",
                  file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
