# ouro-python

An agent that grows its own harness inside a stock Python interpreter.

Inspired by **Dominik Tornow's [Ouro. The Last Harness](https://x.com/i/article/2108102766837092353)**
([original X post](https://x.com/DominikTornow/status/2108224785918439600)).
This project ports the concept from his [Emacs/Lisp Ouro repository](https://github.com/dtornow/ouro)
to Python. Credit for the one-prompt, one-eval bootstrap idea belongs to Dominik.

Python **is** the runtime. Python 3.9+ and its standard library are enough.
No packages to install.

`boot.py` supplies HTTP transport, a persistent Python namespace, `eval`, and
usage limits. Its initial `agent` sends one request and executes one tool call;
it does **not** implement a feedback loop or a chat interface. The bootstrap
prompt asks the model to replace `agent` with a real feedback loop and call it.
The default task then asks that agent to build and run a terminal chat. The
generated agent and chat exist in the same interpreter that runs the bootstrap.

## Run

```sh
git clone https://github.com/renerocksai/ouro-python.git
cd ouro-python
```

Set `OPENAI_API_KEY` in your environment or copy `.env.example` to `.env`
and replace the placeholder:

```dotenv
OPENAI_API_KEY=your-key-here
```

Start the bootstrap:

```sh
python3 boot.py
```

The CLI prints where generated code and usage are saved. The model first
installs its agent loop, then builds and starts the terminal chat. Bootstrap
may take several requests. Type at the generated `You:` prompt; the default
task asks for `/quit` or EOF to exit. Start a new process to grow a new runtime.

For the sibling Neovim project's existing credential file:

```sh
python3 boot.py --env-file ../ouro-nvim/.env
```

Or supply a specific task (the model still bootstraps its loop first):

```sh
python3 boot.py 'Use eval to set answer = 6 * 7, verify it, and report the result.'
```

## API limits

The default model is `gpt-6.1-sol` with medium reasoning. `--reasoning high`
is available. Defaults cap the entire process at 8 HTTP attempts, 4,096 output
tokens per response, 16,384 cumulative output tokens, and 120,000 encoded bytes
of input history per request. Output tokens include reasoning. If the remaining
budget is smaller, the next response cap shrinks to match it. Failed attempts
count; the harness does not retry. A truncated response stops before executing
its code. `--max-requests`, `--max-output-tokens`, `--total-output-tokens`,
`--max-input-bytes`, and `--timeout` change those limits. These bound requests
made through `agent_prompt`; generated Python has the runtime's full privileges.

Bootstrap and conversation share the same process budget. The verified run
used all eight default requests for bootstrap and two chat turns. To allow
more conversation, choose a larger request allowance explicitly:

```sh
python3 boot.py --max-requests 16 --total-output-tokens 16384
```

Input tokens are reported in usage traces but not limited by a tokenizer.
Complete histories are resent on each request and billed again; the input
byte limit bounds each request's size. The output budget is not a dollar limit.

## Inspect the generated runtime

Every eval is saved before execution in `.ouro-python/run-*/eval-*.py`. Usage
metadata goes into `usage.jsonl`, and the CLI prints the location and final usage.
`--output-dir PATH` chooses another location. Code and trace files have mode
0600; fresh run directories have mode 0700. Credentials are redacted from saved
code and tool feedback. Complete model response histories, including encrypted
reasoning, stay in memory. These generated files let you inspect what built and
changed the live runtime.
Printed tool output is both returned to the model and displayed immediately,
so generated `input()` prompts and terminal chat output remain visible.

## How it works

The seed supplies `agent_prompt(history)` for HTTP requests,
`agent_eval(code)` for shared-runtime evaluation, and a one-shot `agent`.
The bootstrap includes the seed's own source and asks the model to redefine
the global `agent`, then call that replacement with the user's task.

Generated code executes with `eval` or `exec` in a persistent namespace.
Functions and variables created by one eval are available to the next.
Expressions return their representation; statement chunks can set `_result`.
Printed output and exceptions become tool results for the generated loop.

Generated code runs locally with normal Python access to files and processes.
Use Ctrl+C to interrupt. There is no prebuilt chat, no hardcoded replacement
agent, and no guarantee a model will successfully bootstrap on every attempt.

The transport uses the [Responses API](https://developers.openai.com/api/docs/guides/reasoning)
with `store: false`. The bootstrap prompt requires replaying every output item,
including reasoning and encrypted content, before appending tool results.

## Troubleshooting

- **Waiting during bootstrap:** HTTP requests can take up to the default
  120-second timeout. Generated chat appears after the model installs it.
- **HTTP 503:** the service is overloaded. Start a fresh process to retry;
  there are no automatic retries.
- **Missing API key:** check `OPENAI_API_KEY`, `.env`, or `--env-file PATH`.
- **Request or token limit reached:** start a fresh process with an explicit
  allowance. The generated loop should stop on budget errors.
- **Incomplete response:** partial code is not evaluated. A larger
  `--max-output-tokens` may be needed, especially with high reasoning effort.

## Verify

Run the mocked tests without an API key or network access:

```sh
python3 -m unittest -v
```

They verify that the seed has no loop, generated code installs a working loop,
state persists across evals, tool results and encrypted reasoning reach the next
request, limits stop further calls, and incomplete/error responses stop safely.
A bounded live run with `gpt-6.1-sol` and medium reasoning succeeded: the model
installed the agent loop, built and started a terminal chat, used eval to set
`persistent_value = 42`, recalled `42` on the next turn without another tool,
and exited on `/quit`. A direct assertion against the interpreter's shared
globals confirmed the mutation. The run used 8 API requests and 635 reported
output tokens (5,751 input tokens, of which 3,218 were cached). Generated code
and usage are saved locally and excluded from Git.

The editor-hosted version is [ouro-nvim](https://github.com/renerocksai/ouro-nvim).
