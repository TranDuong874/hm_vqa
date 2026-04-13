# HM-VQA Taxonomy

## Core Abstraction

For HM-VQA, the minimal abstraction is:

- `time`
- `entity`

Everything in video QA can be represented as:
- reasoning over entities
- reasoning over time
- or compositions of the two

This keeps the taxonomy small and makes it easier to design reusable tooling.


## 1. Time

Time is the first core object.

### 1.1 Direct Time

Direct or absolute time references:
- timestamp
- interval
- clip range
- moment

Examples:
- `at 00:35:40`
- `between 00:12:10 and 00:12:20`
- `during the clip from 10s to 15s`

### 1.2 Relative Time

Relative time references:
- before
- after
- during
- between
- nearest to
- first / last / next / previous occurrence

Examples:
- `before kneading dough`
- `after the blender was turned off`
- `what happened 2 minutes earlier`


## 2. Entity

Entity is the second core object.

An entity can be:
- `action`
- `person`
- `object`
- `speech`
- `event`
- `state`
- `attribute`

Examples:
- action: `knead dough`
- person: `the cook`, `left hand`
- object: `bowl`, `lid`, `spatula`
- speech: spoken phrase, sound event
- event: `making pasta sauce`
- state: `lid is on pot`, `bowl is empty`
- attribute: count, location, property

This means we do not need a separate first-class abstraction like `semantic`.
`entity` is enough as the core bucket.


## Query Families

All QA can be written as mappings between `time` and `entity`.

### 1. Time -> Entity

Given a time reference, recover what entities are present or active.

Examples:
- what action happened at `t`?
- who was interacting with the sink at `t`?
- what object was on the table during this interval?

This includes:
- direct time -> entity
- relative time -> entity


### 2. Entity -> Time

Given an entity, localize when it occurred.

Examples:
- when did `turn off blender` happen?
- when was the bowl placed down?
- when did the person speak?

This is the main temporal localization setting.


### 3. Entity -> Entity

Given one entity, infer another.

Examples:
- what was the person doing before kneading dough?
- what object was used after opening the cupboard?
- who was present when the lid was rotated?

In practice, most `Entity -> Entity` queries are implemented as:

- `Entity -> Time -> Entity`

So time is often the latent intermediate even when the query surface looks purely semantic.


### 4. Time + Entity -> Time

Given both a time reference and an entity, refine or constrain the answer in time.

Examples:
- when exactly after `X` did the pouring stop?
- what happened immediately before the lid was placed?


### 5. Multi-hop Compositions

More complex queries chain multiple mappings.

Examples:
- `Entity -> Time -> Entity`
- `Entity -> Time -> Time -> Entity`
- `Time -> Entity -> Time`

These are the cases where explicit tooling becomes most important.


## Operational View

The taxonomy should not stop at query labels.
We need operations that act on `time` and `entity`.

The main operation families are:

- `retrieve`
- `filter`
- `compare`
- `aggregate`
- `verify`

These operations can be defined over:
- time
- entity
- or both jointly


## Tooling Implication
For current MCQ test, how would LLM approach it? 

Example:
When did action <knead dough happen>?
A. At time <>
B. At time <>
...

LLM CoT:
Approach 1:
1. When did the action kneading dough happen -> Localize and retrieve multiple instances: Call entity -> time tool to localize entity -> get list of entities and their time
2. Now we need to look at the mcq choices -> the given choices are in these times
3. LLM filter time, maybe use tool here to filter, like compare_time(choice_A_time, candidate_time), if smaller than threshold or overlap or whatever output.
4. Limit choices, use inspection tool to actually sample and retrieve L1 frames and answer

Approach 2: 
1. Read MCQ first, extract the time, throw into the tool -> retrieve around these times and inspect

Basically, the tooling needs to be general enough so LLM can take both approach, and applicable for other types of reasoning, questions, entities, not just MCQ on localization, and not just MCQ


Chains for different types of questions:
1. when did: E -> T
Example:
"When did/what action, event, object,... happen, exist"

2. entity_temporal_relation: compare(E -> T, E -> T)
"Did A happen before B?"

3. order_of_events: E -> T ×N -> sort
"In which order did A, B, C occur?"

4. before_after: E -> T -> T -> E
"What happened before/after X?"

5. context_at_time: T -> E
"What was I doing at 3pm?"

6. between_events:
E -> T, E -> T -> T -> E
"What happened between A and B?"

7. entity_entity_relation:E -> E
"What was the man doing to the bread?"
E(bread) -> E(action)

"Where was the bread?"
E(bread) -> E(location)