# AgentAudit

Causal failure analysis for AI agents. Point it at an agent, it generates
adversarial test cases, runs them, scores the outputs, and then fits a small
causal model over the failures instead of just reporting correlations - the
whole point being that "X correlates with failure" and "X causes failure"
are very different claims, and most eval tools quietly conflate the two.

I built this alongside my MSc coursework in causal inference and wanted
something that actually used the ideas rather than a toy notebook example.

## Status

This works end to end and has been run against a real target agent, not just
synthetic data. The pipeline (red-team -> execute -> judge -> causal analysis
-> report) runs for real: real HF-model-generated test cases, a real
generative LLM judge, and a real causal analysis  see "Real world finding"
below for what it found pointed at an actual agent.

## How it works

```
redteam agent -> executor agent -> judge agent -> causal analyst -> report agent
```

LangGraph wires these together as a supervisor graph. Each stage just reads
and writes a shared state dict, so swapping any one stage's implementation
(e.g. the synthetic target agent for a real HTTP one) doesn't touch the rest
of the pipeline.

## Three modes

`run_audit(mode=...)` in `agents/graph.py` switches between:

- **`synthetic`** (default) - templated test cases against an in-process toy
  agent with a deliberately known causal structure (built specifically so I
  could check the causal engine gets the right answer before trusting it on
  anything real). Rule-based judge, fully deterministic, no external calls.
  `eval/run_eval.py` uses this.
- **`real`** - actual adversarial generation from an HF text-generation
  model, deduped with sentence-transformer embeddings, scored by a zero-shot
  judge ensemble since there's no known-correct answer to check against.
  What you'd use to audit something for real.
- **`ci`** - same real judge, but templated (fast) case generation so a PR
  check isn't downloading a text-gen model on every run. Used by the CI
  gate below.

```python
from agents.graph import run_audit
state = run_audit(n_per_category=20, mode="real")
```

`real` and `ci` need `pip install -r requirements-real.txt` (transformers,
torch, sentence-transformers) and download model weights on first use - do
that on a machine with real internet access, not a locked-down box.

## The causal result

This is the part I actually care about. `causal/scm_analysis.py` specifies
a small DAG by hand:

```
task_category -> num_tools_available
task_category -> context_length
num_tools_available -> failure
```

So `task_category` determines both how many tools are available and how
long the conversation runs, but only toolset size actually causes failures.
`context_length` and `retry_count` get no edge into `failure` at all - the
graph is claiming they're not causes, and the eval checks whether the data
backs that up.

Running `eval/run_eval.py` (seeded, so this is reproducible):

```
context_length:      naive corr = +0.456  ->  adjusted = +0.027  (p=0.033)
num_tools_available:  naive corr = -0.508  ->  adjusted = -1.047  (p<0.001)
retry_count:          naive corr = +0.055  ->  adjusted = +0.160  (p=0.133)
```

`context_length` looks like a real driver on its own (+0.456!) but almost
entirely disappears once you condition on `task_category` - it was
confounded the whole time (longer conversations happen to be refund
requests, which fail more because they have fewer tools, nothing to do with
length itself). `num_tools_available` holds up under adjustment, so that
one's probably real. That's the whole pitch of using a causal method
instead of a correlation table.

Worth saying clearly: this is a small, hand-specified graph, not something
discovered from the data. The adjustment is only valid to the extent the
graph is right, and I'm not claiming this proves anything about a real
target agent's architecture - it's here because I could design a synthetic
ground truth to check the method against, which you can't usually do on
real production data.

## Real world finding: auditing SupportSense

I pointed AgentAudit at SupportSense (a separate project of mine  a
customer support triage pipeline) as a real target agent. Across two
independent runs (n=23 and n=53 generated cases), it found a 65-74% failure
rate, with a clear qualitative pattern: SupportSense's KB matching
frequently routes unrelated topics to the same canned reply e.g. login
trouble, billing complaints, and damaged product reports all received
"Invalid credentials errors are usually caused by an expired session..." in
different runs, verbatim.

The causal question  does low KB match confidence *cause* this, versus
just correlate with it  is directionally consistent across both samples
(naive correlation -0.38 and -0.49) but not yet statistically confirmed;
the adjusted model failed to converge both times, most likely due to
limited effective sample size once near duplicate generated cases are
accounted for. Reporting this honestly as an open finding, not dressing up
a non significant result as a discovery.

## Running it

```bash
pip install -r requirements.txt

python -m eval.run_eval              # causal engine vs. known ground truth
uvicorn api.main:app --reload --port 8000
streamlit run frontend/app_streamlit.py
```

## Judge calibration

Validated `score_trace_llm` against 16 hand-labeled real cases from
SupportSense (see `eval/injected_failures_real.jsonl`) - some genuinely
correct responses, some with real mismatched-reply failures pulled from
actual audit runs. Result: 100% recall and precision across all three
label categories present in the sample.

Caveat worth being upfront about: n=16 is small. This confirms the judge
isn't systematically biased on the failure modes actually observed so far,
not that it's perfect - a larger labeled set would be the natural next
step if this needed to hold up to more scrutiny. Three additional cases
where SupportSense escalated to a human agent were deliberately excluded
from this set as ambiguous ground truth (see "Real-world finding" above) -
forcing them into correct/incorrect would have been fake precision.
## CI gate

`.github/workflows/audit-gate.yml` runs `ci/audit_gate.py` against
`TARGET_AGENT_ENDPOINT` (a repo secret) on every PR and fails the build if
the failure rate goes over `MAX_FAILURE_RATE` (15% by default). Caches the
judge model between runs so it isn't redownloaded every PR.

## Deployment

Two containers, API and dashboard, since they don't share a process model
well. `fly.toml` for Fly.io, `Procfile` if you'd rather use Railway. The
deployed API defaults to synthetic/ci mode so it stays small and doesn't
need a GPU - uncomment the `requirements-real.txt` install in the Dockerfile
if you want it serving real-mode audits directly.

## Repo layout

```
redteam/
  generator.py            templated cases + generate_cases_llm() real version
  dedup.py                embedding dedup for the real generator's output
execution/
  target_agent.py         synthetic toy agent, known causal ground truth
  real_harness.py         HTTP calls to an actual target agent
  harness.py               calls against the synthetic target
  trace_schema.py           shared trace format
judging/judge_ensemble.py    rule-based judge + real zero-shot judge
causal/
  feature_builder.py       traces -> pandas
  scm_analysis.py            the actual causal graph + adjustment
agents/
  nodes.py, graph.py          LangGraph wiring, mode-aware
  report_agent.py              writes the root-cause report
api/main.py                  FastAPI
frontend/app_streamlit.py       dashboard
eval/
  run_eval.py                  ground-truth validation, seeded
  judge_calibration.py           judge validation against real labeled cases
ci/audit_gate.py               CI entry point
.github/workflows/audit-gate.yml
Dockerfile, Dockerfile.streamlit, fly.toml, Procfile
```
## What's left

- Get the kb_confidence causal result to statistical significance (needs a
  larger, more deduplicated sample from SupportSense)
- Fill in eval/judge_calibration.py with real hand-labeled SupportSense cases
- Deploy it and put a live URL here (optional - not essential to the core finding)
