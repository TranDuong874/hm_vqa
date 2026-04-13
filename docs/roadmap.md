# Paper Arc

## Working Claim

Long-video video QA should not rely on a monolithic "retrieve then answer" pipeline.
It benefits from explicit tool-like operations over temporal evidence, candidate sets, and frame selection.

Our current evidence supports this more strongly than a claim that hierarchical retrieval alone beats strong VLM baselines on end-to-end MCQ.


## Core Observation

Pure VLM baselines can achieve competitive or better MCQ accuracy while remaining weakly grounded.

Our retrieval-grounded variants:
- improve evidence relevance
- improve grounded conversion when the selected clip is correct
- reduce "correct despite miss" behavior

But they still fail mainly at:
- selecting the correct instance among repeated nearby candidates

So the bottleneck is not just retrieval score. It is the lack of explicit temporal operations inside the retrieval-to-answer pipeline.


## Proposed Framing

Instead of presenting the work as only a new hierarchical retriever, frame it as:

- tool-augmented video retrieval and reasoning
- standard operations for video query examination
- explicit temporal and evidence operations that make LLM involvement in retrieval more structured

This better matches the current results and generalizes beyond a single benchmark.


## Proposed Tool Layer

Keep the tool inventory small and reusable.

### 1. TemporalNormalize

Normalize timestamps from:
- questions
- MCQ options
- retrieved segments
- sampled frames

Output:
- structured intervals in one canonical format

### 2. TemporalFilter

Filter candidate clips using temporal compatibility with the query or MCQ choices.

Output:
- compatible candidates
- rejected candidates

### 3. TemporalDistance

Compute:
- overlap
- nearest-gap distance
- ordering consistency

This makes temporal reasoning explicit instead of implicit inside prompting.

### 4. CandidateCompare

Given a shortlist of candidate clips:
- compare them
- identify the strongest evidence clip
- explain why weaker candidates should be rejected

### 5. EvidenceSelect

Within a candidate clip:
- retrieve or sample diverse informative keyframes
- avoid redundant consecutive frames

### 6. ConsistencyCheck

Given:
- selected evidence
- proposed answer

verify whether the answer is temporally consistent with the shown evidence.

### 7. EvidencePack

Package the final selected evidence into a structured bundle for the answerer:
- selected candidate
- frame timestamps
- optional temporal metadata


## Pipeline Story

The paper should present a pipeline with explicit operations:

1. Candidate harvest
- retrieve coarse/fine candidates

2. Temporal filtering
- prune candidates with explicit temporal operations

3. Candidate selection
- choose the best candidate from the shortlist

4. Evidence selection
- select the most informative frames inside the chosen clip

5. Final answer
- answer from selected evidence only

6. Optional consistency check
- reject temporally inconsistent answers


## Main Experimental Story

### 1. Motivation

End-to-end VLM QA often gets answers correct without tightly relevant evidence.

MCQ accuracy alone hides this problem.

### 2. Method

Introduce a tool layer for video query examination:
- temporal normalization
- temporal filtering
- candidate comparison
- evidence selection
- answer consistency checking

### 3. Findings

The current results support:
- stronger grounding
- better evidence precision
- cleaner hit-to-answer conversion

The remaining major error is:
- repeated-instance disambiguation after shortlist

### 4. Implication

Video QA systems need explicit operations over time and evidence, not just better similarity search or larger end-to-end prompts.


## Eval Matrix

The evaluation story should be:

- baseline vs tooling
- across query families
- using the smallest broad pilot that can reject bad directions early

Primary pilot comparison:

- `pure_vlm`
- `tooling`

Ablations are deferred until tooling shows value.
Significance is deferred until the tooling direction is stable.

### HD-EPIC 8-task Pilot

| Task | Query Family | Why It Matters For HM-VQA | Likely Transfer |
| --- | --- | --- | --- |
| `fine_grained_action_localization` | `Entity->Time` | Hard repeated-instance temporal grounding; stress test for explicit time tools | LongVideoBench, MLVU |
| `recipe_step_localization` | `Entity->Time` | Multi-video temporal grounding and longer instructional steps | LongVideoBench |
| `fine_grained_action_recognition` | `Time->Entity` | Tests whether retrieved memory supports action understanding beyond timestamp selection | Video-MME, MLVU |
| `recipe_step_recognition` | `Time->Entity` | Event/step understanding in instructional videos | LongVideoBench, Video-MME |
| `ingredient_ingredients_order` | `Entity->Entity` | Explicit temporal order reasoning over entities and events | LongVideoBench, MLVU |
| `object_motion_object_movement_counting` | `Time->Entity` | Counting over time; tests memory aggregation and repeated-event tracking | Video-MME, MLVU |
| `fine_grained_how_recognition` | `Entity->Entity` | Procedural explanation; tests whether memory supports descriptive reasoning | Video-MME, LongVideoBench |
| `fine_grained_why_recognition` | `Entity->Entity` | Intent/causal reasoning grounded in local video evidence | Video-MME, LongVideoBench |

### Pilot Policy

- compare only `pure_vlm` vs `tooling`
- sample `30` examples per task
- use broad task coverage first, significance later
- promote tooling if it ties or slightly beats baseline on MCQ while improving grounding, interpretability, or error structure

### Pilot Size

- target size: `240` examples
- local-video-only
- stratified by:
  - video length bucket
  - localization hardness where available
- capped to avoid one-video concentration within a task

### Fairness Policy

- cheap screening run:
  - `pure_vlm@16`
  - `tooling`
- fairness audit only after a promising pilot:
  - rerun `pure_vlm@24` on the same split


## Metrics To Emphasize

Do not report MCQ accuracy alone.

Report:
- MCQ accuracy
- candidate pool hit-any
- shortlist hit-any
- selected evidence hit@1
- answer accuracy given selected hit
- answer accuracy given selected miss
- TP / FN / FP / TN style grounded counts
- tolerance-based hit rates:
  - exact
  - +-5s
  - +-10s
  - +-30s


## Honest Positioning

Do not claim:
- hierarchy alone beats strong VLM baselines on end-to-end MCQ

Do claim:
- explicit video-query operations improve grounding
- they make LLM involvement in retrieval more structured and more faithful
- benchmark scores and grounding quality can diverge significantly


## Concrete Next Steps

1. Keep the current split-final-stage + thinking + L1-keyframe variant as the promoted grounded baseline.
2. Build a broad HD-EPIC pilot split across the 8-task eval matrix.
3. Compare only `pure_vlm` vs `tooling` on the pilot.
4. Promote tooling only if it ties or improves MCQ while improving grounding or interpretability.
5. Run a fairness audit against `pure_vlm@24` only after tooling shows value on the pilot.
