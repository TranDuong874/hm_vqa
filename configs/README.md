# Experiment Configs

The experiment stack is split into four config types:

- `datasets/`: where examples, videos, features, and results live.
- `memory/`: how reusable offline video memory is constructed.
- `retrieval/`: how existing memory is searched and converted into evidence.
- `answer/`: which VLM backend answers from the selected evidence.

This keeps ablations out of dataset loaders. Dataset code should only load
examples and compute benchmark-specific metrics; memory/retrieval/answer choices
come from these configs or equivalent CLI flags.

The current runners still keep their historical CLI flags for compatibility.
New runners should prefer config files and write the resolved config into each
run directory as `config.json`.

