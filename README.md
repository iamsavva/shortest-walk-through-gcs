# Installation

1. install drake
2. install `shortest_walk_through_gcs` and its dependencies. From inside this repository, run the following:
```
pip install -e .
```


## known issues and TODO
- ellipsoids for convex sets don't seem to work --- there is a bug that result in the cost-to-go just being constant
- use_skill_compoisition_constraint_add hacky option needs to be removed and fixed
- need to add proper parallelization of MathematicalPrograms solves, or alternatively use cvxpy
- remove vertex is start? shouldn't matter; i don't think i use that flag at all
- add a check to ensure program is not built twice
- remove assertion about adding vertex into target, instead print a warning sign or something