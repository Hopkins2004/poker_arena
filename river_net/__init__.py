"""River value network PoC (DeepStack style) on top of the exact river solver.

Modules: :mod:`cards` (combo tables), :mod:`tree` / :mod:`solver` (exact river solver, in
Rust via ``pokerarena._native``), :mod:`ranges` (training-spot generator), :mod:`buckets`,
:mod:`datagen`, :mod:`dataset`, :mod:`model`, :mod:`train`, :mod:`evaluate`.
See :mod:`river_net.solver` for the value convention.
"""
