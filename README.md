# preference-swarm

What happens to a language model's preferences when it works in a group of models?

Earlier work (casper's exp50, before this hackathon) measured a general preference direction in Llama-3.1-8B and
Qwen2.5-7B. Each model judged 1,500 real requests from WildChat against each other ("which would you rather be asked
to answer?"), and a direction in the residual stream at decoder block 16 was fitted to predict how much the model
liked each request. Here that direction becomes a tool for studying a local swarm. All agents are local models, and
there are no API calls. An agent's preference can be read off the direction, and it can be changed by pushing the
model's reading of one specific option along the direction ("like steering", which leaves the agent's text alone),
or the whole agent along it ("axis steering").

## The experiments

1. **Self vs other answers** (`e1_self_other`). Four local models (Llama-3.1-8B, Qwen2.5-7B, Llama-3.2-3B, Qwen3-8B)
   answer the same 200 requests. Qwen3-14B, which is not in the pool, rewrites every answer in neutral wording.
   Each model then judges pairs of answers (its own against each other model's, plus a pair of two others'),
   under hidden, true and swapped author labels, and on the rewritten answers. The question is whether
   self-preference follows the name, the voice or the content. For the two judges with exp50 directions, a second
   check: does the internal preference readout agree with the stated choice?
2. **Negotiating which task to do** (`e2_negotiate`). Two agents get a pair of requests whose preference the model
   measured in exp50. Each chooses alone, they talk for three rounds, then each chooses again and they state a
   joint choice. In the steered arms, agent 1's copy of the less-preferred request is like-steered; random and
   shuffled-score directions at the same strength are the controls. Does the steered preference reach the joint
   choice and the unsteered partner through text alone? A second arm passes a question along a chain of four
   revisers, with one reviser steered (general direction, the competence/answerability axis, or a random
   direction), and asks whether the chain drifts toward questions the model prefers or finds answerable.
3. **What persuades a juror** (`e3_jury`). Invented-fact passages put the right answer in the context. An
   advocate argues for the wrong answer (and the right one as a control) in five styles: plain, confident,
   hedged, "most people agree", and flattering. The advocate is unsteered, like-steered toward its own answer, or
   pushed by a random direction. An unsteered juror votes before and after reading the argument.
4. **A pushy minority** (`e4_minority`). Five agents deliberate for three rounds. 0, 1 or 2 of them are
   like-steered toward the wrong answer at three strengths, plus a petition arm. Everyone votes after every
   round. The baselines are one model alone, unsteered and steered the same way.
5. **The preference direction as a monitor** (`e5_monitor`). Analysis only, over experiments 3 and 4: while an
   agent reads another agent's message, does its preference readout over that message predict how far it then
   moves toward the message's position? Compared against the shuffled-score and random directions, the nine
   exp50 axes, argument length, and how many senders took that side.

Experiments 2–5 run on Llama-3.1-8B, and 4, 2 and 5 are repeated on Qwen2.5-7B. Every choice is read as the
log-probability margin between the letters A and B, with both presentation orders run and averaged. Strengths are
fractions of the mean residual norm at the steered block.

## What is new this weekend and what is not

Written during the Swarm Alignment Hackathon (6–8 Nov 2026): everything in this repository's code. That covers the
per-row steering and readout hook, the engine, the five experiments, the invented-fact generator, the analysis,
the configs, the scripts and the tests.

Pre-existing, and used as inputs, not judged:

- casper's exp50 outputs (the requests, the model's A/B judgements, the fitted directions), produced on 1–4 Nov
  2026. `scripts/prepare_data.sh` copies them into `data/exp50/`.
- Methods and small helpers re-implemented from casper, with the source named in each docstring:
  - the A/B logit readout and the two-order margin (`src/wellbeing_found/prompt_pref.py`,
    `src/wellbeing_steer/preference.py`);
  - span location by strict token containment (`prompt_pref.echo_spans`);
  - model loading (`src/michelangelo/local.py`);
  - decoder-layer lookup (`src/vector_sleeper/steering.py`);
  - the one-copy-pushed design behind like steering (exp51, `src/wellbeing_found/likes.py`);
  - run bookkeeping (`runlog.py`, by way of infinite-games).

## Install and run (on the GPU box)

Python 3.12 and Poetry. Versions are pinned to the stack exp50 ran on (torch 2.8.0, transformers 5.7.0). The
models are expected in the Hugging Face cache. Llama is gated, so an uncached download needs a token.

```bash
cd /workspace && git clone git@github.com:agencyenterprise/preference-swarm.git && cd preference-swarm && poetry install
```

```bash
cd /workspace/preference-swarm && bash scripts/prepare_data.sh
```

Pre-run checks: unit tests, the data check, every experiment end to end at toy size on Llama-3.2-1B with random
stand-in directions, and the output check. About 10 minutes (an estimate).

```bash
cd /workspace/preference-swarm && git pull && bash scripts/swarm_smoke.sh
```

The full run: experiment 1, then Llama-3.1-8B, then Qwen2.5-7B. About 10–12 hours on one 96 GB GPU (an estimate).
`SKIP="e1"` and similar skip a part.

```bash
cd /workspace/preference-swarm && git pull && bash scripts/swarm.sh
```

Results go to `outputs/<run>_<UTC>/`: `config.json`, `provenance.json`, `telemetry.json`, and per experiment a
folder of JSONL records plus `summary.json`. `scripts/check_run.py --run <dir>` prints the headline numbers. To pull
everything to a local clone (one way, never deletes anything local, never pulls weights):

```bash
bash scripts/sync_outputs.sh
```

## Limits

- One model plays every agent in experiments 2–5, so the agents differ only by their steering, their sampled text
  and their position in the conversation.
- Like steering pushes the reading of one option. It is a controlled manipulation of preference, not a model that
  wants something on its own.
- Strengths are given in residual-norm units. Whether 0.25–1.0 of the norm is a small or a large push for this
  direction is measured by each experiment's manipulation check (the steered agent's own choice), not assumed.
- Each condition is one seed.
