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

from pydrake.math import (  # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)

from queue import PriorityQueue

from shortest_walk_through_gcs.program_options import ProgramOptions

from shortest_walk_through_gcs.util import (  # pylint: disable=import-error, no-name-in-module, unused-import
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
    add_set_membership,
) 

# from gcs_util import get_edge_name, make_quadratic_cost_function_matrices, plot_a_gcs

from shortest_walk_through_gcs.gcs_dual import PolynomialDualGCS
from shortest_walk_through_gcs.solve_restriction import  RestrictionSolution
from shortest_walk_through_gcs.incremental_search import get_name_repeats
from shortest_walk_through_gcs.util_gcs_specific import get_edge_name


def triple_integrator_postprocessing(
                                    graph: PolynomialDualGCS,
                                    options: ProgramOptions,
                                    convex_set_path: T.List[ConvexSet],
                                    vertex_name_path: T.List[str],
                                    vel_bounds: ConvexSet,
                                    acc_bounds: ConvexSet,
                                    jerk_bounds: ConvexSet,
                                    start_state:npt.NDArray,
                                    target_state:npt.NDArray,
                                    cost_function: T.Callable,
                                    traj_reversed:bool,
                                    delta_t:float,
                                    ratio:float,
                                    past_solution=None,
                                    )-> T.Tuple[RestrictionSolution, T.List[float], float]:
    unique_vertex_names, repeats = get_name_repeats(vertex_name_path)

    INFO("using double integrator post-processing", verbose = options.verbose_restriction_improvement)
    
    schedules = []
    num = len(unique_vertex_names)
    for i in range(2**num):
        pick_or_not = bin(i)[2:]
        if len(pick_or_not) < num:
            pick_or_not = "0"*(num - len(pick_or_not)) + pick_or_not

        delta_t_schedule = [delta_t] * (len(convex_set_path)-1)
        for index, pick in enumerate(pick_or_not):
            if pick == "1":
                delta_t_schedule[sum(repeats[:index])] = delta_t * ratio
        schedules.append(np.array(delta_t_schedule))
    

    solve_times = [0.0]*len(schedules)
    que = PriorityQueue()
    for i, schedule in enumerate(schedules):
        x_traj_sol, v_traj_sol, a_traj_sol, j_traj_sol, cost, solve_time = solve_triple_integrator_convex_restriction(
            graph,
            options,
            convex_set_path,
            vertex_name_path,
            vel_bounds,
            acc_bounds,
            jerk_bounds,
            start_state,
            target_state,
            schedule,
            cost_function,
            traj_reversed,
            past_solution,
            )
        solve_times[i] = solve_time
        if x_traj_sol is not None:
            que.put((cost + np.random.uniform(0,1e-9), (x_traj_sol, v_traj_sol, a_traj_sol, j_traj_sol, schedule)))

    if options.use_parallelized_solve_time_reporting:
        num_parallel_solves = np.ceil(len(solve_times)/options.num_simulated_cores_for_parallelized_solve_time_reporting)
        total_solver_time = np.max(solve_times)*num_parallel_solves
        INFO(
            "double inegrator postprocessing, num_parallel_solves",
            num_parallel_solves,
            verbose = options.verbose_restriction_improvement
        )
        INFO("each solve time", np.round(solve_times, 3), verbose = options.verbose_restriction_improvement)
    else:
        total_solver_time = np.sum(solve_times)


    if que.empty():
        WARN(
            "double integrator no improvement",
            verbose = options.verbose_restriction_improvement
        )
        return None, None, None, None, None, total_solver_time
    
    best_cost, (x_traj_sol, v_traj_sol, a_traj_sol, j_traj_sol, schedule) = que.get()


    INFO(
        "triple integrator time improvement from",
        delta_t * (len(vertex_name_path)-1),
        "to",
        np.sum(schedule),
        verbose = options.verbose_restriction_improvement
    )
    INFO(
        "triple inegrator postprocessing time",
        total_solver_time,
        verbose = options.verbose_restriction_improvement
    )

    return x_traj_sol, v_traj_sol, a_traj_sol, j_traj_sol, schedule, total_solver_time



def double_integrator_postprocessing(
                                    graph: PolynomialDualGCS,
                                    options: ProgramOptions,
                                    convex_set_path: T.List[ConvexSet],
                                    vertex_name_path: T.List[str],
                                    vel_bounds: ConvexSet,
                                    acc_bounds: ConvexSet,
                                    start_state:npt.NDArray,
                                    target_state:npt.NDArray,
                                    cost_function: T.Callable,
                                    traj_reversed:bool,
                                    delta_t:float,
                                    ratio:float,
                                    past_solution=None
                                    )-> T.Tuple[RestrictionSolution, T.List[float], float]:
    
    unique_vertex_names, repeats = get_name_repeats(vertex_name_path)

    INFO("using double integrator post-processing", verbose = options.verbose_restriction_improvement)
    
    schedules = []
    num = len(unique_vertex_names)-1
    for i in range(2**num):
        pick_or_not = bin(i)[2:]
        if len(pick_or_not) < num:
            pick_or_not = "0"*(num - len(pick_or_not)) + pick_or_not

        delta_t_schedule = [delta_t] * (len(convex_set_path)-1)
        for index, pick in enumerate(pick_or_not):
            if pick == "1":
                delta_t_schedule[sum(repeats[:index])] = delta_t * ratio
        schedules.append(np.array(delta_t_schedule))
    
    # schedules = [options.delta_t * np.ones(len(vertex_name_path)-1)]

    solve_times = [0.0]*len(schedules)
    que = PriorityQueue()
    for i, schedule in enumerate(schedules):
        # ERROR(schedule)
        # ERROR(vertex_name_path)
        # ERROR(start_state, target_state)
        # ERROR("---")
        x_traj_sol, v_traj_sol, a_traj_sol, cost, solve_time = solve_double_integrator_convex_restriction(
            graph,
            options,
            convex_set_path,
            vertex_name_path,
            vel_bounds,
            acc_bounds,
            start_state,
            target_state,
            schedule,
            cost_function,
            traj_reversed,
            # past_solution
            )
        solve_times[i] = solve_time
        if x_traj_sol is not None:
            que.put((cost + np.random.uniform(0,1e-9), (x_traj_sol, v_traj_sol, a_traj_sol, schedule)))

    if options.use_parallelized_solve_time_reporting:
        num_parallel_solves = np.ceil(len(solve_times)/options.num_simulated_cores_for_parallelized_solve_time_reporting)
        total_solver_time = np.max(solve_times)*num_parallel_solves
        INFO(
            "double inegrator postprocessing, num_parallel_solves",
            num_parallel_solves,
            verbose = options.verbose_restriction_improvement
        )
        INFO("each solve time", np.round(solve_times, 3), verbose = options.verbose_restriction_improvement)
    else:
        total_solver_time = np.sum(solve_times)


    if que.empty():
        WARN(
            "double integrator no improvement",
            verbose = options.verbose_restriction_improvement
        )
        return None, None, None, None, total_solver_time
    
    best_cost, (x_traj_sol, v_traj_sol, a_traj_sol, schedule) = que.get()


    INFO(
        "double integrator time improvement from",
        delta_t * (len(vertex_name_path)-1),
        "to",
        np.sum(schedule),
        verbose = options.verbose_restriction_improvement
    )
    INFO(
        "double inegrator postprocessing time",
        total_solver_time,
        verbose = options.verbose_restriction_improvement
    )

    return x_traj_sol, v_traj_sol, a_traj_sol, schedule, total_solver_time




def solve_triple_integrator_convex_restriction(
    graph: PolynomialDualGCS,
    options: ProgramOptions,
    convex_set_path: T.List[ConvexSet],
    vertex_name_path: T.List[str],
    vel_bounds: ConvexSet,
    acc_bounds: ConvexSet,
    jerk_bounds: ConvexSet,
    start_state: npt.NDArray,
    target_state:npt.NDArray,
    dt: T.List[float],
    cost_function: T.Callable,
    traj_reversed:bool,
    past_solution = None,
):
    """
    solve a convex restriction over a vertex path
    return cost of the vertex_path
    and return a list of bezier curves
    where bezier curve is a list of numpy arrays (vectors).
    """
    # construct an optimization problem
    prog = MathematicalProgram()

    # previous direction of motion -- for bezier curve continuity
    x_traj = []
    v_traj = []
    a_traj = []
    j_traj = []

    # ERROR([convex_set.ambient_dimension() for convex_set in convex_set_path])

    # for every vertex:
    for i, convex_set in enumerate(convex_set_path):
        x = prog.NewContinuousVariables(convex_set.ambient_dimension(), "x"+str(i))
        v = prog.NewContinuousVariables(vel_bounds.ambient_dimension(), "v"+str(i))
        a = prog.NewContinuousVariables(acc_bounds.ambient_dimension(), "a"+str(i))
        x_traj.append(x)
        v_traj.append(v)
        a_traj.append(a)
        if past_solution is not None:
            prog.AddLinearConstraint(eq(x, past_solution[0][i]))
            prog.AddLinearConstraint(eq(v, past_solution[1][i]))
            prog.AddLinearConstraint(eq(a, past_solution[2][i]))

        if i == 0:
            prog.AddLinearEqualityConstraint(x, start_state)
            prog.AddLinearEqualityConstraint(v, np.zeros(vel_bounds.ambient_dimension()))
            prog.AddLinearEqualityConstraint(a, np.zeros(acc_bounds.ambient_dimension()))
        elif i == len(convex_set_path)-1:
            prog.AddLinearEqualityConstraint(x, target_state)
            prog.AddLinearEqualityConstraint(v, np.zeros(vel_bounds.ambient_dimension()))
            prog.AddLinearEqualityConstraint(a, np.zeros(acc_bounds.ambient_dimension()))
        else:
            add_set_membership(prog, convex_set, x, True)
            add_set_membership(prog, vel_bounds, v, True)
            add_set_membership(prog, acc_bounds, a, True)

        if not traj_reversed and i > 0:
            edge_name = get_edge_name(vertex_name_path[i-1], vertex_name_path[i])
            if edge_name not in graph.edges:
                edge_name = get_edge_name(vertex_name_path[i], vertex_name_path[i-1])
            if graph.edges[edge_name].right_edge_point_must_be_inside_intersection_of_left_and_right_sets:
                add_set_membership(prog, convex_set_path[i-1], x, True)
        elif traj_reversed and i+1 <= len(convex_set_path)-1:
            edge_name = get_edge_name(vertex_name_path[i+1], vertex_name_path[i])
            if edge_name not in graph.edges:
                edge_name = get_edge_name(vertex_name_path[i], vertex_name_path[i+1])
            if graph.edges[edge_name].right_edge_point_must_be_inside_intersection_of_left_and_right_sets:
                add_set_membership(prog, convex_set_path[i+1], x, True)



        if i > 0:
            j = prog.NewContinuousVariables(jerk_bounds.ambient_dimension(), "j"+str(i))
            j_traj.append(j)
            if past_solution is not None:
                prog.AddLinearConstraint(eq(j, past_solution[3][i-1]))
                
            add_set_membership(prog, jerk_bounds, j, True)

            cost = cost_function(np.hstack((x_traj[i-1],v_traj[i-1],a_traj[i-1])), j, np.hstack((x_traj[i],v_traj[i],a_traj[i])), target_state, dt[i-1])
            prog.AddCost(cost)
            
            prog.AddLinearConstraint(eq(x_traj[i], x_traj[i-1] + v_traj[i-1]*dt[i-1] + a_traj[i-1] * dt[i-1]**2/2 + j_traj[i-1]*dt[i-1]**3/6 ))
            prog.AddLinearConstraint(eq(v_traj[i], v_traj[i-1] + a_traj[i-1]*dt[i-1] + j_traj[i-1]*dt[i-1]**2/2))
            prog.AddLinearConstraint(eq(a_traj[i], a_traj[i-1] + j_traj[i-1]*dt[i-1]))


    if options.policy_solver is None:
        solution = Solve(prog)
    else:
        if options.policy_solver == MosekSolver:
            mosek_solver = MosekSolver()
            solver_options = SolverOptions()
            # set the solver tolerance gaps
            
            solver_options.SetOption(
                MosekSolver.id(),
                "MSK_DPAR_INTPNT_TOL_INFEAS",
                options.MSK_DPAR_INTPNT_TOL_INFEAS,
            )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
            # )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_PFEAS,
            # )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_DFEAS,
            # )

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
        else:
            solution = options.policy_solver().Solve(prog)

    if options.policy_solver == MosekSolver or solution.get_solver_id().name() == "Mosek":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == ClarabelSolver or solution.get_solver_id().name() == "Clarabel":
        solver_solve_time = solution.get_solver_details().solve_time
    elif options.policy_solver == GurobiSolver or solution.get_solver_id().name() == "Gurobi":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == OsqpSolver or solution.get_solver_id().name() == "OSQP":
        solver_solve_time = solution.get_solver_details().solve_time
    else:
        WARN("don't know how to get solver time for the solver", solution.get_solver_id().name())
        raise NotImplementedError()
    

    if solution.is_success():
        x_traj_sol = np.array([solution.GetSolution(x) for x in x_traj])
        v_traj_sol = np.array([solution.GetSolution(v) for v in v_traj])
        a_traj_sol = np.array([solution.GetSolution(a) for a in a_traj])
        j_traj_sol = np.array([solution.GetSolution(j) for j in j_traj])
        return x_traj_sol, v_traj_sol, a_traj_sol, j_traj_sol, solution.get_optimal_cost(), solver_solve_time
    else:
        return None, None, None, None, np.inf, solver_solve_time
    



def solve_double_integrator_convex_restriction(
    graph: PolynomialDualGCS,
    options: ProgramOptions,
    convex_set_path: T.List[ConvexSet],
    vertex_name_path: T.List[str],
    vel_bounds: ConvexSet,
    acc_bounds: ConvexSet,
    start_state: npt.NDArray,
    target_state:npt.NDArray,
    dt: T.List[float],
    cost_function: T.Callable,
    traj_reversed:bool,
    past_solution = None,
) -> T.Tuple[T.List[RestrictionSolution], float]:
    """
    solve a convex restriction over a vertex path
    return cost of the vertex_path
    and return a list of bezier curves
    where bezier curve is a list of numpy arrays (vectors).
    """
    # construct an optimization problem
    prog = MathematicalProgram()

    # previous direction of motion -- for bezier curve continuity
    x_traj = []
    v_traj = []
    a_traj = []

    # ERROR([convex_set.ambient_dimension() for convex_set in convex_set_path])

    # for every vertex:
    for i, convex_set in enumerate(convex_set_path):
        x = prog.NewContinuousVariables(convex_set.ambient_dimension(), "x"+str(i))
        v = prog.NewContinuousVariables(vel_bounds.ambient_dimension(), "v"+str(i))
        x_traj.append(x)
        v_traj.append(v)
        if past_solution is not None:
            prog.AddLinearConstraint(eq(x, past_solution[0][i]))
            prog.AddLinearConstraint(eq(v, past_solution[1][i]))

        if i == 0:
            prog.AddLinearEqualityConstraint(x, start_state)
            prog.AddLinearEqualityConstraint(v, np.zeros(vel_bounds.ambient_dimension()))
        elif i == len(convex_set_path)-1:
            prog.AddLinearEqualityConstraint(x, target_state)
            prog.AddLinearEqualityConstraint(v, np.zeros(vel_bounds.ambient_dimension()))
            # add_set_membership(prog, Hyperrectangle(-1e-5*np.ones(7), 1e-5*np.ones(7)), v, True)
        else:
            add_set_membership(prog, convex_set, x, True)
            add_set_membership(prog, vel_bounds, v, True)

        if not traj_reversed and i > 0:
            edge_name = get_edge_name(vertex_name_path[i-1], vertex_name_path[i])
            if edge_name not in graph.edges:
                edge_name = get_edge_name(vertex_name_path[i], vertex_name_path[i-1])
            if graph.edges[edge_name].right_edge_point_must_be_inside_intersection_of_left_and_right_sets:
            # if options.add_right_point_inside_intersection_constraint and vertex_name_path[i-1] != vertex_name_path[i]:
            #     YAY(convex_set_path[i-1].IntersectsWith(convex_set_path[i]))
            #     WARN(convex_set_path[i-1].Intersection(convex_set_path[i]).PointInSet(past_solution[0][i]))
                add_set_membership(prog, convex_set_path[i-1], x, True)
        elif traj_reversed and i+1 <= len(convex_set_path)-1:
            edge_name = get_edge_name(vertex_name_path[i+1], vertex_name_path[i])
            if edge_name not in graph.edges:
                edge_name = get_edge_name(vertex_name_path[i], vertex_name_path[i+1])
            # if options.add_right_point_inside_intersection_constraint and vertex_name_path[i] != vertex_name_path[i+1]:
            if graph.edges[edge_name].right_edge_point_must_be_inside_intersection_of_left_and_right_sets:
            #     YAY(convex_set_path[i-1].IntersectsWith(convex_set_path[i]))
            #     WARN(convex_set_path[i-1].Intersection(convex_set_path[i]).PointInSet(past_solution[0][i]))
                add_set_membership(prog, convex_set_path[i+1], x, True)



        if i > 0:
            a = prog.NewContinuousVariables(acc_bounds.ambient_dimension(), "a"+str(i))
            a_traj.append(a)
            if past_solution is not None:
                prog.AddLinearConstraint(eq(a, past_solution[2][i-1]))
                
            add_set_membership(prog, acc_bounds, a, True)

            # if options.add_right_point_inside_intersection_constraint and vertex_name_path[i-1] != vertex_name_path[i]:
            #     YAY(convex_set_path[i-1].IntersectsWith(convex_set_path[i]))
            #     WARN(convex_set_path[i-1].Intersection(convex_set_path[i]).PointInSet(past_solution[0][i]))

            #     add_set_membership(prog, convex_set_path[i-1], x, True)

            cost = cost_function(np.hstack((x_traj[i-1],v_traj[i-1])), a, np.hstack((x_traj[i],v_traj[i])), target_state, dt[i-1])
            prog.AddCost(cost)
            
            prog.AddLinearConstraint(eq(x_traj[i], x_traj[i-1] + v_traj[i-1]*dt[i-1] + a_traj[i-1] * dt[i-1]**2 / 2 ))
            prog.AddLinearConstraint(eq(v_traj[i], v_traj[i-1] + a_traj[i-1]*dt[i-1]))


    if options.policy_solver is None:
        solution = Solve(prog)
    else:
        if options.policy_solver == MosekSolver:
            mosek_solver = MosekSolver()
            solver_options = SolverOptions()
            # set the solver tolerance gaps
            
            solver_options.SetOption(
                MosekSolver.id(),
                "MSK_DPAR_INTPNT_TOL_INFEAS",
                options.MSK_DPAR_INTPNT_TOL_INFEAS,
            )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
            # )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_PFEAS,
            # )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_DFEAS,
            # )

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
        else:
            solution = options.policy_solver().Solve(prog)

    if options.policy_solver == MosekSolver or solution.get_solver_id().name() == "Mosek":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == ClarabelSolver or solution.get_solver_id().name() == "Clarabel":
        solver_solve_time = solution.get_solver_details().solve_time
    elif options.policy_solver == GurobiSolver or solution.get_solver_id().name() == "Gurobi":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == OsqpSolver or solution.get_solver_id().name() == "OSQP":
        solver_solve_time = solution.get_solver_details().solve_time
    else:
        WARN("don't know how to get solver time for the solver", solution.get_solver_id().name())
        raise NotImplementedError()
    

    if solution.is_success():
        # YAY("problem solved")
        x_traj_sol = np.array([solution.GetSolution(x) for x in x_traj])
        v_traj_sol = np.array([solution.GetSolution(v) for v in v_traj])
        a_traj_sol = np.array([solution.GetSolution(a) for a in a_traj])
        return x_traj_sol, v_traj_sol, a_traj_sol, solution.get_optimal_cost(), solver_solve_time
    else:
        # diditwork(solution)
        # from IPython.display import Markdown, display
        # for con in solution.GetInfeasibleConstraints(prog):
        #     display(Markdown(con.ToLatex()))
        return None, None, None, np.inf, solver_solve_time
    