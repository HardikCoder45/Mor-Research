# Recursion and Parameter Sharing Synthesis

## Thesis
Recursive Transformers are one of the most interesting ways to make a small **stored** model behave like a deeper computation graph. They are not automatically faster on a phone. The correct question is whether the extra logical transformations improve quality per resident byte **after accounting for repeated weight reads, cache growth, routing and compiler control flow**.

## 1. Why naive recursion is insufficient
Three bodies of evidence line up:
- older parameter-sharing work shows that one fully shared block is too restrictive and that Cycle/Sequence schedules matter;
- Relaxed Recursive Transformers recovers performance with depth-specific low-rank adapters and good initialization;
- RecursiveVLM shows that feeding a terminal hidden state back into the same computation creates representation mismatch and that a learned normalized connector is useful.

Therefore recursion needs **specialization across logical depth** even if the large matrices are shared.

## 2. EdgeRec-482 design
EdgeRec is our highest-upside research candidate, not the production default.

Physical backbone:
- 25 physical layers;
- d=1280, dFF=3328;
- 20 Q / 4 KV heads, head dimension 64;
- 49,152 tied vocabulary;
- all layers have GQA;
- total learned parameters **482.098M**, including router/connector/adapters.

Logical recurrence:
- preserve unique early/late computation;
- designate a six-layer middle span as recursively reusable;
- default maximum extra recursion = **one** additional pass for selected tokens (logical R≤2 for that span);
- use a causal token-choice/budget router rather than server-style expert-choice as the deployable form.

Learned recurrence overhead (~1.287M total):
- four intermediate taps;
- RMS-normalized low-rank connector with bottleneck 128;
- zero-initialized output/residual path so recurrence starts as a no-op;
- small router `1280→128→3` for depth/budget classes;
- tiny recursion/depth embeddings;
- rank-16 depth adapters on repeated layers.

## 3. Routing policy
A phone-friendly router should make one **up-front** decision per token or per small token group. Re-running top-k selection at each depth is harder to compile and may force dynamic gathers/scatters.

Recommended training budgets:
- economy: no extra pass;
- balanced: target ~25–35% tokens for one extra middle pass;
- quality: target ~50% tokens;
- never route 100% by default because that collapses the adaptive model into a slow fixed recursive model.

The router should receive hidden-state features plus an optional request-level compute budget embedding. Training should expose all budgets so inference does not depend on a threshold never seen during training.

## 4. Monotonic objective
For token `i`, shallow loss `l_i^0` and deeper loss `l_i^1`, add an extra penalty where the deeper computation is worse:

`L_mon = mean_i [ l_i^1 + λ * 1(l_i^1 > l_i^0) * (l_i^1 - l_i^0) ]`.

A smoother hinge/softplus implementation is preferable to a hard indicator in code. The total objective combines:
- next-token CE at the shallow exit;
- next-token CE at the deep exit;
- monotonic penalty;
- router supervision/regularization;
- load/budget penalty to keep average routed fraction near target;
- optional teacher distillation at both exits.

This turns recurrence into an anytime-quality mechanism: extra compute is explicitly trained to have positive expected marginal value.

## 5. Cache choices
### Recursion-wise cache
Store deeper K/V only for routed tokens. This reduces both cache bytes and deeper attention work. It is the preferred latency-quality research mode if the runtime can consume compacted routed tokens efficiently.

### First-pass shared KV
Reuse first-pass K/V for deeper queries. It minimizes additional stored cache, but deeper queries still attend/read the original cache and can remain bandwidth-bound. Use as the RAM/TTFT mode.

## 6. Colab proxy evidence
The proxy EdgeRec implementation actually reused physical middle blocks and routed 50% of tokens through an additional pass. On the T4 test:
- it trained stably for 150 matched steps with finite gradients;
- final validation loss was ~2.347, second only to LocalGlobal (~2.341) in the tiny proxy;
- median 256-token prefill was ~14.67 ms, much slower than FlashSkip (~5.85 ms);
- peak CUDA allocation was ~277 MiB, highest of the five proxies;
- the router auxiliary loss decreased from ~0.73 to ~0.64 but was far from solved.

**Interpretation:** recurrence is trainable and potentially quality-efficient, but the proxy confirms its extra control/computation is real. This supports ranking EdgeRec as a research challenger rather than the initial production architecture.

## 7. Full-size deployment risks
1. Dynamic token gather/scatter can break NPU/ANE delegation.
2. Reused middle weights may be reread from DRAM each logical pass.
3. Branching by token is less efficient at batch 1 than a regular dense graph.
4. Router errors can waste compute on easy tokens or starve hard tokens.
5. Any extra pass increases latency even though stored weights stay fixed.
6. Recursive KV semantics complicate cache kernels.

## 8. What would make EdgeRec beat FlashSkip
Promote EdgeRec to the production winner only if a full ablation demonstrates all of the following at matched W4 resident bytes:
- clear held-out/per-benchmark quality gain at the same training tokens;
- balanced mode increases p50 decode latency by ≤20% versus FlashSkip while providing a material reasoning/knowledge gain;
- economy mode matches or beats FlashSkip latency with acceptable quality;
- routing stays delegated on at least one target phone runtime;
- no pathological p95 tail from dynamic token counts;
- deeper recursion improves hard-token losses rather than merely reprocessing arbitrary tokens.

Until those gates are passed, recursive computation is an **R&D option with high upside**, not the most evidence-backed default.
