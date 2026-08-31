# Independent orchestration reference

home.py and life.py are verbatim copies of the corresponding runtime files at
a8c82ff2b83d0fa50a46323b6443c656d9a79a03, before the LangGraph refactor.
They are loaded only by tests/parity_support.py (which redirects the life
inheritance import to the frozen home executor). Never import them in production.

The parity tests replay synthetic provider results and compare full requests,
state/results and progress/call ordering. Do not update this reference to make a
refactor pass. Shared content/schema/ledger/prompt modules must not change in this
refactor; their existing regression tests remain enabled.
