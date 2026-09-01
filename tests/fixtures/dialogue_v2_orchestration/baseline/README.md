# Independent orchestration reference

home.py and life.py preserve the sequential orchestration of the corresponding
runtime files at a8c82ff2b83d0fa50a46323b6443c656d9a79a03, before the LangGraph
refactor.  Accepted post-refactor semantic contracts are ported to both the
production graph runtime and this sequential reference; graph orchestration is
never copied into this fixture.
They are loaded only by tests/parity_support.py (which redirects the life
inheritance import to the frozen home executor). Never import them in production.

The parity tests replay synthetic provider results and compare full requests,
state/results and progress/call ordering. Do not update this reference merely to
make an orchestration refactor pass.  When an independently approved semantic
change is made, update the sequential implementation explicitly and keep the
existing regression tests enabled.
