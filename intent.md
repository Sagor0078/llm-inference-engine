# Intent: Muse Spark Open Kernel Challenge

This is a Markdown project-intent brief saved as `intent.m` as requested. It is not MATLAB code or an official challenge submission format.

## Objective

Develop correct, reproducible GPU optimizations that make Muse Spark Open faster than stock SGLang on the challenge hardware. Begin with a focused kernel problem, use measured results to guide improvements, and carry useful optimizations into the serving track when available. Optimize real computation that can be upstreamed; never optimize around the evaluator.

## Source of truth

Reviewed on 2026-10-26:
- https://openkernelchallenge.dev/
- https://openkernelchallenge.dev/llms.txt
- Rules to review before participation: https://openkernelchallenge.dev/rules
- Terms to review before participation: https://openkernelchallenge.dev/terms.pdf

Before implementation, refresh the official instructions and read the selected problem's generated TASK.md and AGENTS.md. Those specifications determine signatures, supported shapes, tolerances, scoring and allowed operations. Treat the strategy below as a proposal, not an official requirement.

## Initial scope and choice

Start with `muse_bf16_proj` as a manageable first target. Inspect its shape distribution and baseline before committing to an optimization approach. Consider `muse_moe_w4a16` next if profiling shows stronger opportunities for serving performance. Other available problems are `muse_attention_decode`, `muse_decode_layer`, and `muse_prefill_layer`.

Kernel success means passing every correctness check and achieving a repeatable official speedup greater than 1.0 over SGLang. No unmeasured speedup is a result. Track per-shape regressions as well as the aggregate score.

Kernel boards provide a route to serving priority; prize money is awarded on the serving board. Do not treat a kernel ranking as a prize-winning result.

## Setup workflow

1. Inspect the local environment, GPU availability and installed `okc` version.
2. If needed, obtain the official CLI using the installation instructions on the challenge site.
3. Have the participant run `okc login` themselves to review and accept the rules and complete authentication. Do not accept agreements on their behalf or expose tokens in logs.
4. Run `okc problems` and `okc show muse_bf16_proj` to check current availability and limits.
5. Run `okc init muse_bf16_proj` in a dedicated working directory.
6. Read TASK.md, AGENTS.md, submission.py, baseline.py, reference.py and checks.py before changing code.
7. Establish a correct baseline and record the environment, source revision and timings.

The current request is for this brief only. Installing software, authenticating, running competition jobs and submitting entries are future execution steps.

## Kernel implementation contract

- Deliver one Python submission file defining the required `kernel_fn` interface and preserve the problem identifier expected by the CLI.
- Target one NVIDIA H200 SXM and the official SGLang image. Use only dependencies available there; the judge has no network access.
- Use the problem's untimed `prepare` hook for permitted weight preparation where provided. Do not move input-dependent computation out of the timed call.
- Keep execution CUDA-graph compatible: no host synchronization or data-dependent host control flow. Dispatch on documented shapes and configuration only.
- Preserve mandated precision and model semantics. Weight repacking must be lossless. Any precision exception must be explicitly allowed by the current problem rules.
- Preserve full required attention coverage and exact routing semantics. Never skip keys, experts or operations to exploit tolerances.
- Do not modify the evaluator, cache or replay outputs, inspect hidden expected outputs, or manipulate timing.

## Optimization process

For each candidate:
1. Identify a bottleneck using the baseline and available profiling evidence.
2. Write a specific hypothesis: what work or data movement changes, why it should help, and which shapes may regress.
3. Make one coherent change. Candidate techniques include tiling, launch configuration, weight layout, legal fusion and reducing redundant memory traffic.
4. Validate outputs against the official reference across required shapes and fresh random inputs. Include nearby shapes and edge cases supported by the contract.
5. Benchmark with the official timing approach and compare the full shape distribution. Record compilation and preparation costs separately where relevant.
6. Use `okc check submission.py` for an official unscored check when useful; inspect its result before spending a scored attempt.
7. Retain improvements supported by repeatable evidence and revert regressions. Keep a known-good candidate.

Read current quotas from `okc problems`; do not hard-code a submission budget. Record each remote job ID and inspect its terminal result. Avoid duplicate submissions and reserve attempts for promising candidates.

## Serving phase

When the serving track is open and the participant proceeds:
- Initialize the pinned SGLang source with `okc serve init` and read its generated patch rules.
- Transfer validated improvements into the real serving path and profile at the official workload loads.
- Produce the patch using `okc serve diff`; restrict changes to permitted source and JSON files under `python/sglang/`.
- Respect exclusions, patch-size limits and all model, precision, drafter and output requirements from the current rules.
- Validate with `okc serve check` before a scored run when feasible.
- Evaluate throughput at stock SGLang's interactivity, correctness, and time to first token across the official loads. A faster isolated kernel is insufficient evidence of a serving improvement.

## Deliverables

- A compliant `submission.py` for each selected kernel problem.
- A concise experiment log containing hypothesis, revision, correctness outcome, per-shape timings, aggregate speedup and official job IDs.
- Reproduction instructions identifying the environment and exact commands.
- A best-known validated candidate and a clear account of remaining bottlenecks.
- For the serving phase, a compliant `patch.diff` plus validation evidence and a short explanation suitable for upstream review.

## Completion criteria

The implementation phase is complete when the deliverable passes official correctness checks, shows reproducible improvement against the official baseline, obeys current rules, and can be reproduced from the recorded source and commands. Distinguish local measurements, official check results, scored results and provisional leaderboard positions. If hardware or quotas prevent validation, explicitly report the unverified work rather than claiming success.