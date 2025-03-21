import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    GurobiSolver,
    MosekSolverDetails,
    SnoptSolver,
    OsqpSolver,
    ClarabelSolver,
    IpoptSolver,
    SolverOptions,
    CommonSolverOption,
)
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
)

from pydrake.symbolic import (  # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)
from pydrake.math import (  # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)

from shortest_walk_through_gcs.program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

from shortest_walk_through_gcs.util import ( # pylint: disable=import-error, no-name-in-module, unused-import
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
    latex
)  

from shortest_walk_through_gcs.util_gcs_specific import get_edge_name

from shortest_walk_through_gcs.gcs_dual import PolynomialDualGCS, DualEdge, DualVertex

from shortest_walk_through_gcs.util import add_set_membership, recenter_convex_set

class RestrictionSolution:
    """
    A purpose-built class for storing restrictions over a vertex sequence
    """
    def __init__(self, vertex_path:T.List[DualVertex], trajectory:T.List[npt.NDArray], edge_variable_trajectory:T.List[npt.NDArray] = None):
        self.vertex_path = vertex_path
        self.trajectory = trajectory
        self.edge_variable_trajectory = edge_variable_trajectory 
        if self.edge_variable_trajectory is None:
            self.edge_variable_trajectory = [None] * (len(vertex_path)-1)
        assert len(self.edge_variable_trajectory) == len(vertex_path)-1

        self.expanded_subpath = None

    def vertex_now(self) -> DualVertex:
        return self.vertex_path[-1]
    
    def point_now(self) -> npt.NDArray:
        return self.trajectory[-1]
    
    def point_initial(self) -> npt.NDArray:
        return self.trajectory[0]
    
    def edge_var_now(self) -> npt.NDArray:
        return self.edge_variable_trajectory[-1] if len(self.edge_variable_trajectory) > 0 else None
    
    def length(self):
        return len(self.trajectory)

    def vertex_names(self):
        return np.array([v.name for v in self.vertex_path])
    
    def make_copy(self):
        return RestrictionSolution(self.vertex_path, self.trajectory, self.edge_variable_trajectory)

    def extend(self, next_point: npt.NDArray, next_edge_var: npt.NDArray, next_vertex: DualVertex) -> "RestrictionSolution":
        if not next_vertex.convex_set.PointInSet(next_point):
            # point not in set, need to project due to bad numerics
            next_point = next_vertex.convex_set.Projection(next_point)[1].flatten()

        return RestrictionSolution(
                    self.vertex_path + [next_vertex],
                    self.trajectory + [next_point], 
                    self.edge_variable_trajectory + [next_edge_var]
                    )
    
    def reoptimize(self, graph:PolynomialDualGCS, verbose_failure:bool, target_state:npt.NDArray, one_last_solve:bool):
        new_sol, solver_time = solve_convex_restriction(graph, self.vertex_path, self.point_initial(), verbose_failure, target_state, one_last_solve)
        assert new_sol is not None, WARN("resolving on the same restriction failed. something is fishy")
            
        self.trajectory = new_sol.trajectory
        self.edge_variable_trajectory = new_sol.edge_variable_trajectory
        return solver_time

    def get_cost(self, graph:PolynomialDualGCS, 
                 use_edge_cost_surrogate: bool, 
                 add_target_heuristic:bool = True, 
                 target_state:npt.NDArray = None,
                 ) -> float:
        """
        Note: this is cost of the full path with the target cost
        """
        if target_state is None:
            if graph.options.potentials_are_not_a_function_of_target_state:
                target_state = np.zeros(graph.target_convex_set.ambient_dimension())
            else:
                assert isinstance(graph.target_convex_set, Point), "target set not passed when target set not a point"
                target_state = graph.target_convex_set.x()

        cost = 0.0
        for index in range(len(self.trajectory)-1):
            edge = graph.edges[get_edge_name(self.vertex_path[index].name, self.vertex_path[index + 1].name)]
            x_v = self.trajectory[index]
            x_v_1 = self.trajectory[index+1]
            u = self.edge_variable_trajectory[index]
            if use_edge_cost_surrogate:
                cost += edge.cost_function_surrogate(x_v, u, x_v_1, target_state)
            else:
                cost += edge.cost_function(x_v, u, x_v_1, target_state)                

        if add_target_heuristic:
            cost_to_go_at_last_point = self.vertex_path[-1].get_cost_to_go_at_point(self.trajectory[-1], target_state, graph.options.check_cost_to_go_at_point)
            cost += cost_to_go_at_last_point
        return cost


def solve_parallelized_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_paths: T.List[T.List[DualVertex]],
    state_now: npt.NDArray,
    verbose_failure=True,
    target_state:npt.NDArray = None,
    one_last_solve:bool = False,
    verbose_solve_success = True,
    warmstart:RestrictionSolution = None,
) -> T.Tuple[T.List[RestrictionSolution], float]:
    """
    THIS IS VERY STUPID PARALLELIZATION and should not actually be used.
    what it does is instead of solving n progs, it creates a single big prog and solves it.
    it's stupid.
    proper parallelization is not actually implemented yet;
    TODO: now that SolveInParallel is in drake, should reimplement.

    Returns:
        T.Tuple[T.List[RestrictionSolution], float]: list of RestrictionSolutions and total solve time.
    """

    def add_cons(prog:MathematicalProgram, formulas, con_type:str, linear:bool, verbose=False):
        """
        A simple helper function for quickly adding constraints
        """
        if isinstance(formulas, Expression):
            formulas = [formulas]
        if verbose:
            for formula in formulas:
                print(formula >= 0)
        if con_type == "ge":
            if linear:
                prog.AddLinearConstraint(ge(formulas,0))
            else:
                prog.AddConstraint(ge(formulas,0))
        elif con_type == "eq":
            if linear:
                prog.AddLinearConstraint(eq(formulas,0))
            else:
                prog.AddConstraint(eq(formulas,0))
        else:
            raise NotImplementedError()
        

    options = graph.options

    # construct an optimization problem
    prog = MathematicalProgram()
    vertex_trajectories = []
    edge_variable_trajectories = []
    timer = timeit()

    if target_state is None:
        if options.potentials_are_not_a_function_of_target_state:
            target_state = np.zeros(graph.target_convex_set.ambient_dimension())
        else:
            if isinstance(graph.target_convex_set, Point):
                # assert , "target set not passed when target set not a point"
                target_state = graph.target_convex_set.x()
            else:
                target_state = prog.NewContinuousVariables(graph.target_convex_set.ambient_dimension())

    for _, vertex_path in enumerate(vertex_paths):
        # previous direction of motion -- for bezier curve continuity
        vertex_trajectory = []
        edge_variable_trajectory = []

        # for every vertex:
        for i, vertex in enumerate(vertex_path):
            x = prog.NewContinuousVariables(vertex.state_dim, "x"+str(i))
            vertex_trajectory.append(x)
            if warmstart is not None:
                if i < warmstart.length():
                    prog.SetInitialGuess(x, warmstart.trajectory[i])
                else:
                    # NOTE: this is heuristic. we don't have a guess for these variables
                    prog.SetInitialGuess(x, warmstart.trajectory[-1])
            
            if not vertex.vertex_is_target:
                add_set_membership(prog, vertex.convex_set, x, True)
            else:
                assert i == len(vertex_path) - 1, "something is fishy"

                if options.potentials_are_not_a_function_of_target_state:
                    add_set_membership(prog, vertex.convex_set, x, True)
                else:
                    # new implementation
                    if options.relax_target_condition_during_rollout and not one_last_solve:
                        # during rollout: relax target condition.
                        # at the end when solving restriction -- don't.
                        assert vertex.relaxed_target_condition_for_policy is not None
                        terminating_condition = recenter_convex_set(vertex.relaxed_target_condition_for_policy, target_state)
                        add_set_membership(prog, terminating_condition, x, True)
                    else:
                        prog.AddLinearConstraint(eq(x, target_state))

                # NOTE: this might be helpeful if i ever want to define terminal state to be a point and relax final conditions
                # if options.relax_target_condition_during_rollout and not one_last_solve:
                #     # during rollout: relax target condition.
                #     # at the end when solving restriction -- don't.
                #     assert vertex.relaxed_target_condition_for_policy is not None
                #     terminating_condition = recenter_convex_set(vertex.relaxed_target_condition_for_policy, target_state)
                #     add_set_membership(prog, terminating_condition, x, True)
                # else:
                #     if options.potentials_are_not_a_function_of_target_state:
                #         add_set_membership(prog, vertex.convex_set, x, True)
                #     else:
                #         prog.AddLinearConstraint(eq(x, target_state))



            if i == 0:
                # NOTE: if state_now is None, we have a free initial state problem
                if state_now is not None:
                    prog.AddLinearEqualityConstraint(x, state_now)
            else:
                edge = graph.edges[get_edge_name(vertex_path[i - 1].name, vertex.name)]

                if edge.right_edge_point_must_be_inside_intersection_of_left_and_right_sets and edge.left.state_dim == edge.right.state_dim:
                    if edge.left.name != edge.right.name:
                        add_set_membership(prog, edge.left.convex_set, x, True)

                u = None if edge.u is None else prog.NewContinuousVariables(len(edge.u), "u" + str(i))
                if u is not None and warmstart is not None and len(warmstart.edge_variable_trajectory) > 0:
                    if i < warmstart.length():
                        prog.SetInitialGuess(u, warmstart.edge_variable_trajectory[i-1])
                    else:
                        # NOTE: this is heuristic. we don't have a guess for these variables
                        prog.SetInitialGuess(u, warmstart.edge_variable_trajectory[-1])


                if one_last_solve:
                    for evaluator in edge.one_last_solve_linear_inequality_evaluators:
                        add_cons(prog, evaluator(vertex_trajectory[i-1], u, x, target_state), "ge", True)

                    for evaluator in edge.one_last_solve_equality_evaluators:
                        add_cons(prog, evaluator(vertex_trajectory[i-1], u, x, target_state), "eq", True)

                edge_variable_trajectory.append(u)
                if edge.u_bounding_set is not None:
                    add_set_membership(prog, edge.u_bounding_set, u, True)

                
                cost = edge.cost_function(vertex_trajectory[i-1], u, x, target_state)
                prog.AddCost(cost)

                for evaluator in edge.linear_inequality_evaluators:
                    add_cons(prog, evaluator(vertex_trajectory[i-1], u, x, target_state), "ge", True)

                for evaluator in edge.equality_evaluators:
                    add_cons(prog, evaluator(vertex_trajectory[i-1], u, x, target_state), "eq", True)

                # TODO: in practice this should be put as lorentz cone, not quadratic
                # THIS DEPENDS
                # SHOULD ADD NON-CONVEX OPTION
                for evaluator in edge.quadratic_inequality_evaluators + edge.convex_quadratic_inequality_evaluators:
                    add_cons(prog, evaluator(vertex_trajectory[i-1], u, x, target_state), "ge", False)

                # groebner bases related stuff
                for evaluator in edge.groebner_basis_equality_evaluators:
                    add_cons(prog, evaluator(vertex_trajectory[i-1], u, x, target_state), "eq", True)
                    # prog.AddLinearConstraint(eq(evaluator(vertex_trajectory[i-1], u, x, target_state), 0))

            # add heuristic cost on the last point
            if i == len(vertex_path) - 1:  
                if not options.policy_use_zero_heuristic:
                    cost = vertex.get_cost_to_go_at_point(x, target_state, options.check_cost_to_go_at_point)
                    prog.AddCost(cost)

        vertex_trajectories.append(vertex_trajectory)
        edge_variable_trajectories.append(edge_variable_trajectory)

    timer.dt("just building", print_stuff=options.verbose_solve_times)
    
    if options.policy_solver is None:
        solution = Solve(prog)
    else:
        if options.policy_solver == MosekSolver:
            mosek_solver = MosekSolver()
            solver_options = SolverOptions()
            # set the solver tolerance gaps
            if not one_last_solve:
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
                )
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_PFEAS,
                )
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_DFEAS,
                )
            solver_options.SetOption(
                MosekSolver.id(),
                "MSK_DPAR_INTPNT_TOL_INFEAS",
                options.MSK_DPAR_INTPNT_TOL_INFEAS,
            )

            solver_options.SetOption(MosekSolver.id(), 
                                    "MSK_IPAR_PRESOLVE_USE", 
                                    options.MSK_IPAR_PRESOLVE_USE)
            
            solver_options.SetOption(MosekSolver.id(), 
                                        "MSK_IPAR_INTPNT_SOLVE_FORM", 
                                        options.MSK_IPAR_INTPNT_SOLVE_FORM)
            
            if options.MSK_IPAR_OPTIMIZER is not None:
                solver_options.SetOption(MosekSolver.id(), 
                                            "MSK_IPAR_OPTIMIZER", 
                                            options.MSK_IPAR_OPTIMIZER)
                
                
            # solve the program
            solution = mosek_solver.Solve(prog, solver_options=solver_options)
        elif options.policy_solver == SnoptSolver:
            if not one_last_solve:
                prog.SetSolverOption(SnoptSolver().solver_id(), "Major iterations limit", options.policy_snopt_major_iterations_limit)
                prog.SetSolverOption(SnoptSolver().solver_id(), "Minor iterations limit", options.policy_snopt_minor_iterations_limit)
                prog.SetSolverOption(SnoptSolver().solver_id(), "Minor feasibility tolerance", options.policy_snopt_minor_feasibility_tolerance)
                prog.SetSolverOption(SnoptSolver().solver_id(), "Major feasibility tolerance", options.policy_snopt_major_feasibility_tolerance)
                prog.SetSolverOption(SnoptSolver().solver_id(), "Major optimality tolerance", options.policy_snopt_major_optimality_tolerance)
            solution = options.policy_solver().Solve(prog)
        else:
            solution = options.policy_solver().Solve(prog)

    timed_solve_time = timer.dt("just solving", print_stuff=options.verbose_solve_times)

    if options.policy_solver == MosekSolver or solution.get_solver_id().name() == "Mosek":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == ClarabelSolver or solution.get_solver_id().name() == "Clarabel":
        solver_solve_time = solution.get_solver_details().solve_time
    elif options.policy_solver == GurobiSolver or solution.get_solver_id().name() == "Gurobi":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == OsqpSolver or solution.get_solver_id().name() == "OSQP":
        solver_solve_time = solution.get_solver_details().solve_time
    elif options.policy_solver == SnoptSolver or solution.get_solver_id().name() == "SNOPT":
        solver_solve_time = solution.get_solver_details().solve_time 
    else:
        WARN("don't know how to get solver time for the solver", solution.get_solver_id().name())
        raise NotImplementedError()

    final_result = []
    if solution.is_success():
        for i, v_path in enumerate(vertex_paths):
            vertex_trajectory = vertex_trajectories[i]
            edge_variable_trajectory = edge_variable_trajectories[i]
            traj_solution = [solution.GetSolution(x) for x in vertex_trajectory]
            edge_traj_solution = [None if u is None else solution.GetSolution(u) for u in edge_variable_trajectory]
            final_result.append(RestrictionSolution(v_path, traj_solution, edge_traj_solution))
        return final_result, solver_solve_time
    else:
        WARN("failed to solve", verbose = verbose_solve_success)
        if verbose_failure:
            diditwork(solution)
            latex(prog)
        return None, solver_solve_time
    

def solve_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_path: T.List[DualVertex],
    state_now: npt.NDArray,
    verbose_failure:bool =False,
    target_state:npt.NDArray = None,
    one_last_solve = False,
    warmstart:RestrictionSolution = None,
) -> T.Tuple[RestrictionSolution, float]:
    result, dt = solve_parallelized_convex_restriction(graph, [vertex_path], state_now, verbose_failure, target_state, one_last_solve, False, warmstart)
    if result is None:
        return None, dt
    else:
        return result[0], dt
