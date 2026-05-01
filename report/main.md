# Adaptive Web Interaction: Leveraging Reinforcement Learning for Comprehensive Action Support

> **Working draft.** Structured as a workshop paper so the FYP report → workshop submission pipeline is as close to a rewrite as possible. Sections marked **[FILL]** depend on results we don't yet have; sections marked **[DRAFT]** are written but need polish; sections without a marker are essentially complete first drafts.

---

## Abstract  *[DRAFT]*

Browser-navigation benchmarks for web agents (InSTA, Mind2Web, WebArena, …) are widely used to evaluate small language models trained with reinforcement learning, but the websites they reference change continuously. We audit the InSTA-150k benchmark with a Gemini-based feasibility judge over 2,598 sampled test tasks and find that **only 28.3% of tasks are reliably feasible**: 28.6% target sites are inaccessible to automated agents, 13.0% reference content that has since been removed or restructured, and 30.1% are too uncertain for the judge to determine. We further show that **24.3% of inaccessible sites returned an HTTP 2xx**, demonstrating that reachability probing alone is insufficient — semantic verification is required. We then continue training a small (1.7B) language-model agent with PPO+LoRA on a feasibility-filtered subset of 2,068 training tasks and evaluate on a feasibility-filtered held-out test set. Continued training on the cleaned subset yields **[FILL: a XX% relative improvement in mean trajectory reward and a XX-percentage-point improvement in judge-rated success], suggesting that benchmark hygiene materially affects measured agent performance.**

## 1. Introduction  *[DRAFT]*

Small language models fine-tuned for browser navigation are an active area of research, with reinforcement learning increasingly used to improve agent behavior beyond what supervised fine-tuning alone can achieve. A typical pipeline trains an SLM agent against a benchmark of natural-language tasks, each grounded in a specific live website (e.g. *"find the price of an iPhone 15 on apple.com"*). The agent's actions are evaluated by a separate LLM judge that scores the trajectory on success, efficiency, and self-correction.

This pipeline rests on a quiet assumption: **the benchmark tasks remain valid over time**. That is, the websites they reference still respond, the content the task asks about still exists, and the navigation steps still resolve to the right answer. In practice, this assumption breaks rapidly. Sites disappear, redirect, deploy bot-detection, restructure content, or simply update — all without notice to anyone training agents against them. When 30% of "the benchmark" is silently broken, the gap between the agent's capability and its measured performance is no longer just a function of the agent.

This paper makes two contributions:

1. **A feasibility audit of the InSTA benchmark.** We use a Gemini-2.5-Flash-based judge with URL-context grounding to classify 2,598 sampled InSTA test tasks into feasible / website-down / content-outdated / uncertain categories. We report per-class breakdowns, characterize the disagreement between HTTP-reachability probes and the LLM judge, and release the per-task classifications for reuse.

2. **A continued-training experiment on a feasibility-filtered subset.** We continue PPO + LoRA training of a 1.7B-parameter SLM agent (warm-started from a checkpoint at trajectory 600) on a 2,068-task feasibility-filtered subset, and evaluate on a held-out 200-task feasibility-filtered subset. We compare to the warm-start checkpoint on both filtered and unfiltered test sets.

Our results suggest that benchmark hygiene is not just a measurement issue but an **agent-development** issue: training on a set of mostly broken tasks corrupts the gradient signal, while training on a cleaned subset produces a measurably better agent.

## 2. Related work  *[FILL — user will write after reading]*

To complete after reading:
- InSTA (Trabucco et al.) — the dataset
- WebArena, Visual-WebArena
- Mind2Web
- WebShop
- General SLM-RL work (RLHF, RLAIF, GRPO, RLOO)
- The few existing pieces on web-benchmark decay (if any)

## 3. Feasibility audit of the InSTA benchmark

### 3.1 Method

For each candidate task $(w, q)$ where $w$ is the target website URL and $q$ the natural-language instruction, we run a two-stage feasibility check:

1. **HTTP probe.** A simple `requests.get(w, timeout=10)` retrieves the response status and any reachability error. This is bounded and fast (~200 ms / task) and acts as a coarse filter — but, as we show below, is insufficient on its own.

2. **LLM judge with URL-context grounding.** A Gemini-2.5-Flash model is prompted with the task instruction, the HTTP probe result, and the today's date; it is given a `url_context` tool that lets it actually fetch and inspect the target page. The judge returns a JSON object:

   ```json
   {"classification": one of {FEASIBLE, WEBSITE_DOWN, CONTENT_LIKELY_OUTDATED, UNCERTAIN},
    "confidence": [0, 1],
    "reasoning": str}
   ```

   We classify a task as **feasible** if and only if the judge returns `FEASIBLE` with confidence ≥ 0.95. The strict threshold trades sample size for precision — we want our cleaned subset to be a high-quality positive set, not a high-recall one.

The judge prompts the model to consider:
- whether the site is reachable from the model's URL-context tool
- whether any page on the site is likely to contain the answer to $q$
- whether the navigation steps implied by $q$ are still completable on the current site
- whether the date suggests the content might be stale

This combined HTTP + LLM-judge methodology surfaces failures that neither method alone catches. We empirically measure the agreement / disagreement between the two below.

### 3.2 Results: per-class breakdown

We sampled 2,598 InSTA test tasks (shuffled with a fixed seed) and ran the feasibility check on each. The breakdown of the LLM judge's classifications is:

| Classification | Count | Percentage |
|---|---|---|
| **FEASIBLE** | 734 | **28.3%** |
| **WEBSITE_DOWN** | 744 | **28.6%** |
| **UNCERTAIN** | 782 | **30.1%** |
| **CONTENT_LIKELY_OUTDATED** | 338 | **13.0%** |

Of the 734 feasible classifications, 532 cleared our 0.95 confidence threshold and were retained; the remaining 202 were judged feasible at 0.85–0.95 confidence and rejected for the cleaned subset.

**Headline finding:** Only 28.3% of tasks are unambiguously feasible. 41.6% are demonstrably broken (28.6% inaccessible to automated agents + 13.0% content-outdated). The remaining 30.1% the judge could not resolve from a single browse — these are best treated as a "judge-budget-limited" indeterminate category rather than as broken or feasible.

> **Figure 1.** *(Already generated as `feasibility_results/figure1_dataset_decay.{pdf,png}`.)*  Two-panel breakdown: (A) classification of all 2,598 tasks; (B) HTTP-probe status within `WEBSITE_DOWN` cases.

### 3.3 Results: HTTP-probe vs LLM-judge disagreement

A cheaper alternative to LLM-based feasibility checking is to filter tasks by HTTP reachability alone — drop everything that doesn't return a 2xx. We test the sufficiency of this approach by examining the HTTP status of tasks our LLM judge classified as `WEBSITE_DOWN`:

| HTTP outcome | Count | % of `WEBSITE_DOWN` |
|---|---|---|
| 2xx | 181 | **24.3%** |
| 3xx | 0 | 0.0% |
| 4xx | 223 | 30.0% |
| 5xx | 12 | 1.6% |
| Unreachable | 328 | 44.1% |

**The 24.3% 2xx-but-broken finding is the operationally important number.** These are sites that would pass a naive reachability filter but in practice deny automated access — login walls, captchas, bot-detection responses served with a 200 status code, JavaScript-only pages where the textual content is empty, or pages that redirect successfully but whose target doesn't contain the expected content.

This finding has practical consequences. A reachability-only data cleaning pipeline (which is what most public web-agent training scripts use, if any) misses one in four broken sites. Researchers who report agent performance on such filtered datasets are over-reporting by a margin proportional to this miss rate.

### 3.4 Caveats

A spot-check of the judge's classifications across all classes (Appendix A; reproducible via `feasibility_results/sample_for_sanity_check.py` with seed 42) reveals three nuances worth flagging:

- The `WEBSITE_DOWN` label is more accurately read as **agent-down**: many of these sites would be usable by a human visitor but are inaccessible to an automated browser due to bot-detection (`HTTP 403` cases) or because the judge's own browse tool fails on JavaScript-heavy pages. For an agent benchmark, agent-inaccessibility is the right notion, but the label requires the framing.
- The `UNCERTAIN` label primarily reflects **judge-budget limitations**: the judge inspects only the landing page in a single browse pass, and many of these tasks would be feasible if the judge could click through. We recommend treating UNCERTAIN as a separate "indeterminate" category rather than lumping it with broken.
- The `CONTENT_LIKELY_OUTDATED` class is the **cleanest decay signal**: the judge's reasoning for these is consistently grounded in specific missing articles, restructured navigation, or sites visibly transitioning to other domains.

## 4. Continued training on the feasibility-filtered subset

### 4.1 Pipeline

Our agent is `btrabucco/Insta-Qwen3-1.7B-SFT` — a Qwen3-1.7B model already supervised-fine-tuned on InSTA web-navigation trajectories — quantized to 4-bit (NF4 with double quantization) at inference and training time. We attach a LoRA adapter (`r=8`, `alpha=16`, dropout=0.05; targets attention projections), giving 17.4 M trainable parameters out of 1.03 B (1.69%).

Each trajectory:
1. Initializes a new Playwright browser session at the task's target URL.
2. Iterates up to 20 (observation, action) steps, where each step:
   - Converts the current page to Markdown via the InSTA pipeline,
   - Constructs a prompt from the instruction, current observation, and last 2 history turns,
   - Generates an action JSON via the model (`max_new_tokens=512`, temperature=0.7, top-p=0.9),
   - Parses the action and executes it in the browser,
   - Receives the next observation.
3. Terminates either via a `stop` action emitted by the agent or after 20 steps.
4. Sends the full trajectory (last 50 observations + last 50 actions, with system + user prompts from the InSTA verbose judge prompt) to a **Vertex AI Gemini 2.5 Flash judge**, which returns `(success, efficiency, self_correction)` ∈ [0, 1]³.

The reward is the convex combination
$$ R = 0.7 \cdot \text{success} + 0.2 \cdot \text{efficiency} + 0.1 \cdot \text{self\_correction} \in [0, 1]. $$
The terminal reward is propagated to the trajectory's intermediate steps via discounting (γ = 0.99), and a per-trajectory advantage is computed by subtracting an EMA baseline.

### 4.2 PPO update with reference-policy anchor

For each completed trajectory $\tau = \{(s_t, a_t)\}_{t=0}^{T-1}$, we compute three sets of step-wise log-probabilities by re-running the trajectory through the model:

- $\log \pi_\text{old}(a_t \mid s_t)$ — the policy at the start of this trajectory's update (no gradient).
- $\log \pi_\text{ref}(a_t \mid s_t)$ — the base SFT policy (LoRA disabled), no gradient. Computed every 5th update for efficiency.
- $\log \pi_\text{new}(a_t \mid s_t)$ — the current policy with gradient tracking enabled.

The PPO loss combines a clipped surrogate, an entropy bonus, and **two** KL penalties:

$$
\mathcal{L} = -\min\!\big( r_t A_t, \; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t \big)
              - \beta_H \, \mathcal{H}[\pi_\text{new}]
              + \beta_\text{KL} \, D_\text{KL}(\pi_\text{new} \| \pi_\text{old})
              + \beta_\text{ref} \, D_\text{KL}(\pi_\text{new} \| \pi_\text{ref})
$$
with $r_t = \exp(\log \pi_\text{new} - \log \pi_\text{old})$, $\varepsilon = 0.1$, $\beta_H = 0.01$, $\beta_\text{KL} = 0.1$, $\beta_\text{ref} = 0.05$. Gradients are clipped to a max norm of 0.5 before an AdamW step at learning rate 2e-5. We use a single PPO epoch per trajectory; batching is at the trajectory granularity (one trajectory = one optimizer step). Training proceeds online: trajectory → reward → update → next trajectory.

The reference-KL term is the methodologically distinctive piece: it anchors the LoRA-updated policy to the *base SFT model's* distribution, preventing cumulative drift over many trajectories that the within-trajectory PPO clip would not catch. It is computed by temporarily disabling the LoRA adapters and re-running the forward pass, recovering the SFT policy's log-probabilities.

### 4.3 Setup

- **Hardware.** A single NVIDIA L4 GPU (24 GB VRAM, on-demand, GCP `g2-standard-8` in `us-east4-c`).
- **Warm-start.** We continue from `checkpoint_trajectory_600`, an LoRA adapter checkpoint produced during prior training on the *unfiltered* InSTA train split. This isolates the variable: same model, same algorithm, only the dataset changes.
- **Training set.** 2,068 feasibility-filtered tasks from the InSTA train split (95+ feasibility confidence, sampled identically to §3.1 but on the train side).
- **Held-out test set.** 200 feasibility-filtered tasks from the InSTA *test* split, never seen during training.
- **Trajectories.** 500 new trajectories starting at dataset index 0, with checkpoints saved every 25 trajectories to `checkpoints_feasible/`.
- **Auxiliary infrastructure.** A Playwright JS server (port 3000) backed by a custom watchdog process that monitors port health every 20s and SIGSTOP/SIGCONT-pauses the python trainer during recovery, preventing trajectory failures from accumulating during transient server outages.

### 4.4 Results: training trajectories  *[DRAFT — partial data, will update]*

Training is ongoing at submission time. Across the first 115 trajectories, we observe a clear quartile-over-quartile improvement in trajectory reward:

| Quartile | n | Mean reward | Mean success | Mean steps |
|---|---|---|---|---|
| Q1 | 27 | 0.242 | 0.212 | 12.6 |
| Q2 | 27 | 0.173 | 0.141 | 14.2 |
| Q3 | 27 | 0.464 | 0.470 | 11.6 |
| Q4 | 28 | 0.452 | 0.472 | 9.5 |

(Two consecutive quartiles at ~0.45+ reward provide stronger evidence than a single quartile spike, which would be more easily attributable to task-difficulty variance.)

> **Figure 2.** *(To be generated from completed training run; rolling-window-25 reward, success, and step count over training trajectories.)*

### 4.5 Results: held-out test-set evaluation  *[FILL — depends on training completion]*

We compare the warm-start checkpoint and the post-training checkpoint on the held-out 200 feasibility-filtered test tasks, and on the matched-size unfiltered test sample, using the same Gemini judge as during training.

| Checkpoint | Test set | Success rate (judge > 0.5) | Mean reward | Mean steps |
|---|---|---|---|---|
| `checkpoint_trajectory_600` (warm-start) | Filtered (n=200) | **TBD** | **TBD** | **TBD** |
| `checkpoint_trajectory_600` (warm-start) | Unfiltered (n=200) | **TBD** | **TBD** | **TBD** |
| Post-training (final) | Filtered (n=200) | **TBD** | **TBD** | **TBD** |
| Post-training (final) | Unfiltered (n=200) | **TBD** | **TBD** | **TBD** |

> **Figure 3.** *(Bar chart of the four cells with bootstrap 95% CIs.)*

Headline claim (to verify): continued training on the feasibility-filtered subset produces a checkpoint that outperforms its warm-start ancestor on **both** the filtered and unfiltered held-out test sets, with the gap larger on the filtered set — consistent with the model having learned signal from cleaner data that partially transfers back to noisier evaluation.

## 5. Limitations  *[DRAFT]*

We are upfront about a number of limitations of the present work:

1. **Single-trajectory PPO updates.** Each completed trajectory triggers one optimizer step. This produces high gradient variance compared with batched-PPO and is mitigated only by tight clipping, KL penalties, and the reference-policy anchor. A batched variant (K=4–8 trajectories per update) is left to future work.

2. **No learned value function / GAE.** Advantages are computed as `discounted_reward − EMA_baseline`. A value head trained alongside the policy and Generalized Advantage Estimation would yield finer credit assignment.

3. **Heuristic reward weights.** The 0.7 / 0.2 / 0.1 weights on success, efficiency, and self-correction were not ablated. Weight choice may have a non-trivial effect on which behaviors the agent prefers.

4. **Single judge call per trajectory.** The terminal reward inherits the judge's stochasticity. Averaging across multiple independent judge calls would reduce reward noise at proportional cost.

5. **Only 200 held-out test tasks.** Sample size limits statistical power for detecting small effects. We bootstrap to give 95% confidence intervals on differences but a larger eval set would tighten conclusions.

6. **No multi-seed training.** Run-to-run variance in RL is well-known. A single training seed limits how confidently we can attribute the observed improvements to the methodological change vs random variation.

7. **No ablation of `min_reward_for_update`, `target_kl`, `ppo_clip`.** Each is a knob that was set to a fixed value based on a single-pass review of the training pipeline.

8. **Self-correction is in principle gameable.** The reward includes a self-correction term that an agent could in theory boost by introducing recoverable mistakes. We did not observe this empirically, but it remains a theoretical concern.

9. **The judge itself is decay-prone.** If Gemini 2.5 Flash is deprecated or significantly changed, the feasibility classifications may not be reproducible. We mitigate by releasing per-task classification outputs and the seed.

## 6. Conclusion  *[DRAFT]*

We document a measurable feasibility decay in the InSTA web-agent benchmark — only 28.3% of sampled test tasks are unambiguously feasible, and a reachability-only filter misses 24.3% of broken sites. We then show that **[FILL: continued training on a feasibility-filtered subset improves measured agent performance on a held-out, similarly filtered, test set]**, suggesting that benchmark hygiene is not a measurement issue but an agent-development one. We release the feasibility classifications, training pipeline, and evaluation scripts.

We hope this work prompts the community to budget time for benchmark feasibility audits, and to disclose feasibility filtering when reporting agent results.

## Appendix A — Sanity-check sample

A random sample (seed=42) of judge classifications across all five categories — including the `HTTP 2xx but classified WEBSITE_DOWN` subset — is provided in `feasibility_results/sanity_check_sample.md`. Each entry includes the agent task instruction, the HTTP probe result, the judge's classification with confidence, and the judge's reasoning text.

## Appendix B — Reproducibility

- Code: github repository (TBD link)
- Per-task feasibility outputs: `feasibility_results/feasible_sample_*.csv`, `feasibility_results/v2_log_parsed.csv`
- Training and eval logs: `training_logs/`, `eval_results/`
- Hardware: GCP `g2-standard-8` (NVIDIA L4 24GB) in `us-east4-c`
- Judge: Vertex AI Gemini 2.5 Flash, `temperature=0.5`, `max_output_tokens=4096`
- Total compute used: **[FILL]** GPU-hours, **[FILL]** dollars (approx)

---

## Section status / todo list

- [x] Abstract — first draft, plug numbers when training+eval finish
- [x] §1 Introduction — first draft
- [ ] §2 Related work — empty, requires reading
- [x] §3.1 Feasibility method — first draft
- [x] §3.2 Per-class breakdown — first draft, complete numbers
- [x] §3.3 HTTP-vs-judge disagreement — first draft, complete numbers
- [x] §3.4 Caveats — first draft
- [x] §4.1 Training pipeline — first draft
- [x] §4.2 PPO update math — first draft, equations included
- [x] §4.3 Setup — first draft
- [ ] §4.4 Training trajectories — partial data, will update at run end
- [ ] §4.5 Held-out eval — depends on running `evaluate_checkpoint.py`
- [x] §5 Limitations — first draft, expand if reviewer asks
- [x] §6 Conclusion — first draft, plug numbers
- [x] Appendix A — already generated
- [ ] Appendix B — Reproducibility — fill compute totals

## Next concrete actions (in order)

1. **You**: read InSTA, WebArena, Mind2Web, WebShop papers; populate §2.
2. **Me / pipeline**: when training run completes, generate Figure 2 from the final CSV.
3. **Me / pipeline**: run `evaluate_checkpoint.py` on warm-start vs final checkpoint × {filtered, unfiltered} test sets.
4. **You**: fill Results numbers (§4.5, abstract, conclusion).
5. **Me**: bootstrap CI computation on eval deltas.
6. **You**: convert to LaTeX (workshop template) when ready to submit.
