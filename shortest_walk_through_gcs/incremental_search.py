import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
)

import plotly.graph_objects as go  # pylint: disable=import-error

from queue import PriorityQueue

from shortest_walk_through_gcs.program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

from shortest_walk_through_gcs.util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
    ChebyshevCenter,
    get_kth_control_point
)  # pylint: disable=import-error, no-name-in-module, unused-import

from shortest_walk_through_gcs.gcs_dual import PolynomialDualGCS, DualEdge, DualVertex
from shortest_walk_through_gcs.solve_restriction import solve_convex_restriction, solve_parallelized_convex_restriction, RestrictionSolution


def precompute_k_step_feasible_paths_from_every_vertex(graph: PolynomialDualGCS, lookaheads: T.List[int]):
    assert graph.table_of_feasible_paths is None
    assert graph.value_function_solution is not None
    graph.table_of_feasible_paths = dict()
    for lookahead in lookaheads:
        lookahead_dict = dict()
        for v_name, vertex in graph.vertices.items():
            vertex_paths = graph.get_all_n_step_paths(lookahead, vertex)
            feasible_vertex_paths = []
            for vertex_path in vertex_paths:
                solution, _ = solve_convex_restriction(graph, vertex_path, None, False, None, True, None)
                if solution is not None:
                    feasible_vertex_paths.append([v.name for v in vertex_path])
            lookahead_dict[v_name] = feasible_vertex_paths
        graph.table_of_feasible_paths[lookahead] = lookahead_dict


def get_k_step_optimal_paths(
    graph: PolynomialDualGCS,
    node: RestrictionSolution,
    target_state: npt.NDArray = None,
) -> T.Tuple[PriorityQueue, float]:
    """ 
    do not use this to compute optimal trajectories
    """
    options = graph.options
    if options.solve_shortest_walk_not_path:
        previous_vertex = None
        if node.length() >= 2:
            previous_vertex = node.vertex_path[-2]
        vertex_paths = graph.get_all_n_step_paths( options.policy_lookahead_horizon, node.vertex_now(), previous_vertex)
    else:
        vertex_paths = graph.get_all_n_step_paths_no_revisits( options.policy_lookahead_horizon, node.vertex_now(), node.vertex_path)

    vertex_paths_after_prohibited = []
    for vertex_path in vertex_paths:
        full_path = [v.name for v in  node.vertex_path[:-1] + vertex_path]
        dont_add = False
        for prohibited_subpath in graph.table_of_prohibited_paths:
            if ''.join(map(str, prohibited_subpath)) in ''.join(map(str, full_path)):
                dont_add = True
                break
        if not dont_add:
            vertex_paths_after_prohibited.append(vertex_path)

    vertex_paths = vertex_paths_after_prohibited
    
    if options.solve_shortest_walk_not_path and graph.options.do_not_return_to_previously_visited_vertices:
        new_vertex_paths = []
        vertex_names_in_path_so_far = node.vertex_names()
        for vertex_path in vertex_paths:
            add_this = True
            still_same = True
            for v in vertex_path:
                if v.name != node.vertex_now().name:
                    still_same = False
                if not still_same and v.name in vertex_names_in_path_so_far:
                    add_this = False
                    break
            if add_this:
                new_vertex_paths.append(vertex_path)
        vertex_paths = new_vertex_paths



    # for every path -- solve convex restriction, add next states
    decision_options = PriorityQueue()

    pre_solve_time = 0.0
    
    INFO("options", len(vertex_paths), verbose=options.policy_verbose_choices)
    solve_times = [0.0]*len(vertex_paths)

    for i, vertex_path in enumerate(vertex_paths):
        next_node = None
        if options.policy_optimize_path_so_far_and_K_step and \
            (len(node.vertex_path) <= options.so_far_and_k_step_initials or len(node.vertex_path) % options.so_far_and_k_step_ratio <= 1e-3):
            # reoptimize current and path so far
            warmstart = node if options.policy_use_warmstarting else None
            r_sol, solver_time = solve_convex_restriction(graph, 
                                                          node.vertex_path[:-1] + vertex_path, 
                                                          node.point_initial(), 
                                                          target_state=target_state, 
                                                          one_last_solve=False,
                                                          warmstart=warmstart)
            if r_sol is not None:
                next_node = RestrictionSolution(r_sol.vertex_path[:node.length()+1], 
                                                r_sol.trajectory[:node.length()+1], 
                                                r_sol.edge_variable_trajectory[:node.length()]
                                                )
            

        else:
            r_sol, solver_time = solve_convex_restriction(graph, 
                                                          vertex_path, 
                                                          node.point_now(), 
                                                          target_state=target_state, 
                                                          one_last_solve=False,
                                                          warmstart=None)
            if r_sol is not None:
                next_node = node.extend(r_sol.trajectory[1], r_sol.edge_variable_trajectory[0], r_sol.vertex_path[1]) # type: RestrictionSolution

        solve_times[i] = solver_time
        if next_node is not None:
            next_node.expanded_subpath = " ".join([v.name for v in vertex_path])
            # NOTE that we use r_sol's cost. that's proper.
            cost_of_que_node = r_sol.get_cost(graph, False, not options.policy_use_zero_heuristic, target_state)
            INFO(r_sol.vertex_names(), np.round(cost_of_que_node, 3), verbose=options.policy_verbose_choices)
            decision_options.put( (cost_of_que_node+np.random.uniform(0,1e-9), next_node ))
        else:
            WARN([v.name for v in vertex_path], "failed", verbose=options.policy_verbose_choices)


    if options.use_parallelized_solve_time_reporting:
        num_parallel_solves = np.ceil(len(vertex_paths)/options.num_simulated_cores_for_parallelized_solve_time_reporting)
        total_solver_time = np.max(solve_times)*num_parallel_solves
    else:
        total_solver_time = np.sum(solve_times)

    total_solver_time += pre_solve_time

    INFO("---", verbose=options.policy_verbose_choices)
    return decision_options, total_solver_time


def lookahead_with_backtracking_policy(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> RestrictionSolution:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    # TODO: add a parallelized options
    options = graph.options
    options.vertify_options_validity()
    INFO("running lookahead backtracking", verbose = options.policy_verbose_choices)

    if target_state is None:
        if options.potentials_are_not_a_function_of_target_state:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()

    expanded_subpaths = dict()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    decision_options = [ PriorityQueue() ]
    decision_options[0].put( (0, RestrictionSolution([vertex], [initial_state], [])) )

    decision_index = 0
    found_target = False
    target_node = None
    number_of_iterations = 0

    total_solver_time = 0.0

    while not found_target:
        if decision_index == -1:
            WARN("backtracked all the way back")
            return None, total_solver_time
        if decision_options[decision_index].empty():
            decision_index -= 1
            INFO("backtracking", verbose=options.policy_verbose_choices)
        else:
            node = decision_options[decision_index].get()[1] # type: RestrictionSolution
            if node.expanded_subpath is not None:
                if node.expanded_subpath not in expanded_subpaths:
                    expanded_subpaths[node.expanded_subpath] = 1
                else:
                    expanded_subpaths[node.expanded_subpath] += 1
                if expanded_subpaths[node.expanded_subpath] > options.subpath_expansion_limit:
                    continue


            INFO("at", node.vertex_names(), verbose = options.policy_verbose_choices)

            if node.vertex_now().vertex_is_target:
                found_target = True
                target_node = node
                break

            # heuristic: don't ever consider a point you've already been in
            # stop = False
            # for point in node.trajectory[:-1]:
            #     if len(node.point_now()) == len(point) and np.allclose(node.point_now(), point, atol=1e-3):
            #         stop = True
            #         break
            # if stop:
            #     print("stopping cause in same point")
            #     continue

            # add another priority que if needed (i.e., first time we are at horizon H)
            if len(decision_options) == decision_index + 1:
                decision_options.append( PriorityQueue() )

            # get all k step optimal paths
            que, solve_time = get_k_step_optimal_paths(graph, node, target_state)
            decision_options[decision_index + 1] = que            
            total_solver_time += solve_time

            # stop if the number of iterations is too high
            number_of_iterations += 1
            if number_of_iterations >= options.iteration_termination_limit:
                WARN("exceeded number of forward iterations")
                return None, total_solver_time
        
            decision_index += 1


    if found_target:
        final_solution, solver_time = postprocess_the_path(graph, target_node, initial_state, target_state)
        total_solver_time += solver_time
        return final_solution, total_solver_time
        
    else:
        WARN("did not find path from start vertex to target!")
        return None, total_solver_time


def cheap_a_star_policy(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    options = graph.options
    options.vertify_options_validity()
    INFO("running cheap A*", verbose = options.policy_verbose_choices)

    if target_state is None:
        if options.potentials_are_not_a_function_of_target_state:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    que = PriorityQueue()
    que.put( (0.0, RestrictionSolution([vertex], [initial_state], []) ) )

    found_target = False
    target_node = None # type: RestrictionSolution
    total_solver_time = 0.0
    number_of_iterations = 0

    max_path_length_so_far = 0

    while not found_target:
        if que.empty():
            WARN("que is empty")
            break
        
        node = que.get()[1] # type: RestrictionSolution
        INFO("at", node.vertex_names(), verbose = options.policy_verbose_choices)

        if node.length() > max_path_length_so_far:
            max_path_length_so_far = node.length()
        elif node.length() < max_path_length_so_far - options.a_star_backtracking_limit:
            continue


        # stop if the number of iterations is too high
        number_of_iterations += 1
        if number_of_iterations >= options.iteration_termination_limit:
            WARN("exceeded number of forward iterations")
            return None, total_solver_time

        if node.vertex_now().vertex_is_target:
            YAY("found target", verbose = options.policy_verbose_choices)
            found_target = True
            target_node = node
            break

        # heuristic: don't ever consider a point you've already been in
        stop = False
        for point in node.trajectory[:-1]:
            if len(node.point_now()) == len(point) and np.allclose(node.point_now(), point, atol=1e-3):
                stop = True
                break
        if stop:
            WARN("skipped a point due to", verbose = options.policy_verbose_choices)
            continue # skip this one, we've already been in this particular point


        # get all k step optimal paths
        next_decision_que, solve_time = get_k_step_optimal_paths(graph, node, target_state)
        total_solver_time += solve_time
        while not next_decision_que.empty():
            next_cost, next_node = next_decision_que.get()
            # TODO: fix this cost; need to extend
            if options.policy_optimize_path_so_far_and_K_step:
                que.put( (next_cost+np.random.uniform(0,1e-9), next_node) )
            else:
                que.put( (next_cost+np.random.uniform(0,1e-9) + node.get_cost(graph, False, False, target_state), next_node) )

    
    if found_target:
        final_solution, solver_time = postprocess_the_path(graph, target_node, initial_state, target_state)
        total_solver_time += solver_time
        return final_solution, total_solver_time
        
    else:
        WARN("did not find path from start vertex to target!")
        return None, total_solver_time

  
def postprocess_the_path(graph:PolynomialDualGCS, 
                          restriction: RestrictionSolution,
                          initial_state:npt.NDArray, 
                          target_state:npt.NDArray = None,
                          ) -> T.Tuple[RestrictionSolution, float]:
    options = graph.options
    cost_before = restriction.get_cost(graph, False, not options.policy_use_zero_heuristic, target_state=target_state)
    timer = timeit()
    total_solver_time = 0.0
    # solve a convex restriction on the vertex sequence
    best_restriction = restriction
    best_cost = cost_before

    if options.postprocess_shortcutting_long_sequences:
        INFO("using long sequence shortcut posptprocessing", verbose = options.verbose_restriction_improvement)
        unique_vertex_names, repeats = get_name_repeats(best_restriction.vertex_names())
        unique_vertices = [graph.vertices[name] for name in unique_vertex_names]
        
        for i, r in enumerate(repeats):
            if r >= options.long_sequence_num:
                shortcut_repeats = make_list_of_shortcuts_for_index(repeats, r, i)
                solve_times = [0.0]*len(shortcut_repeats)
                que = PriorityQueue()
                for j, shortcut_repeat in enumerate(shortcut_repeats):
                    vertex_path = repeats_to_vertex_names(unique_vertices, shortcut_repeat)
                    warmstart = None
                    if options.policy_use_warmstarting:
                        warmstart = get_warmstart_for_a_shortcut(best_restriction, repeats, shortcut_repeat, unique_vertices)
                    new_restriction, solver_time = solve_convex_restriction(graph, vertex_path, initial_state, verbose_failure=False, target_state=target_state, one_last_solve = True, warmstart=warmstart)
                    solve_times[j] = solver_time
                    if new_restriction is not None:
                        restriction_cost = new_restriction.get_cost(graph, False, not options.policy_use_zero_heuristic, target_state=target_state)
                        que.put((restriction_cost+np.random.uniform(0,1e-9), new_restriction))
                if not que.empty():
                    best_cost, best_restriction = que.get(block=False)
                    unique_vertex_names, repeats = get_name_repeats(best_restriction.vertex_names())
                    unique_vertices = [graph.vertices[name] for name in unique_vertex_names]
                if options.use_parallelized_solve_time_reporting:
                    num_parallel_solves = np.ceil(len(solve_times)/options.num_simulated_cores_for_parallelized_solve_time_reporting)
                    total_solver_time += np.max(solve_times)*num_parallel_solves
                else:
                    total_solver_time += np.sum(solve_times)

        
    if options.postprocess_via_shortcutting:
        INFO("using shortcut posptprocessing", verbose = options.verbose_restriction_improvement)
        unique_vertex_names, repeats = get_name_repeats(best_restriction.vertex_names())
        unique_vertices = [graph.vertices[name] for name in unique_vertex_names]
        shortcut_repeats = make_a_list_of_shortcuts(repeats, options.max_num_shortcut_steps)
        solve_times = [0.0]*len(shortcut_repeats)
        que = PriorityQueue()
        for i, shortcut_repeat in enumerate(shortcut_repeats):
            vertex_path = repeats_to_vertex_names(unique_vertices, shortcut_repeat)
            warmstart = None
            if options.policy_use_warmstarting:
                warmstart = get_warmstart_for_a_shortcut(best_restriction, repeats, shortcut_repeat, unique_vertices)
            new_restriction, solver_time = solve_convex_restriction(graph, vertex_path, initial_state, verbose_failure=False, target_state=target_state, one_last_solve = True, warmstart=warmstart)
            solve_times[i] = solver_time
            if new_restriction is not None:
                restriction_cost = new_restriction.get_cost(graph, False, not options.policy_use_zero_heuristic, target_state=target_state)
                que.put((restriction_cost+np.random.uniform(0,1e-9), new_restriction))
        if not que.empty():
            best_cost, best_restriction = que.get(block=False)

        if options.use_parallelized_solve_time_reporting:
            num_parallel_solves = np.ceil(len(solve_times)/options.num_simulated_cores_for_parallelized_solve_time_reporting)
            total_solver_time += np.max(solve_times)*num_parallel_solves
            INFO(
                "shortcut posptprocessing, num_parallel_solves",
                num_parallel_solves,
                verbose = options.verbose_restriction_improvement
            )
            INFO(np.round(solve_times, 3), verbose = options.verbose_restriction_improvement)
        else:
            total_solver_time += np.sum(solve_times)
        INFO("shortcut posptprocessing time", total_solver_time, verbose = options.verbose_restriction_improvement)

    elif options.postprocess_by_solving_restriction_on_mode_sequence:
        INFO("using restriction post-processing", verbose = options.verbose_restriction_improvement)
        best_restriction, solver_time = solve_convex_restriction(graph, best_restriction.vertex_path, initial_state, verbose_failure=False, target_state=target_state, one_last_solve = True)
        total_solver_time += solver_time
        best_cost = best_restriction.get_cost(graph, False, not options.policy_use_zero_heuristic, target_state=target_state)
        INFO("shortcut posptprocessing time", total_solver_time, verbose = options.verbose_restriction_improvement)
        

        
    INFO(
        "path cost improved from",
        np.round(cost_before, 2),
        "to",
        np.round(best_cost, 2),
        "; original is",
        np.round((cost_before / best_cost - 1) * 100, 1),
        "% worse",
        verbose = options.verbose_restriction_improvement
    )
    timer.dt("solve times", print_stuff = options.verbose_solve_times)
    return best_restriction, total_solver_time
             

def obtain_rollout(
    graph: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> T.Tuple[RestrictionSolution, float]:
    graph.options.policy_lookahead_horizon = lookahead
    graph.options.vertify_options_validity()
    options = graph.options
    
    if target_state is None:
        if options.potentials_are_not_a_function_of_target_state:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()

    assert len(state) == vertex.convex_set.ambient_dimension(), "provided state " + str(state) + " not of same dim as " + vertex.name +  " which is " + str(vertex.convex_set.ambient_dimension())
    
    if options.use_greedy_with_backtracking_policy:
        restriction, solve_time = lookahead_with_backtracking_policy(graph, vertex, state, target_state)
    elif options.use_a_star_with_limited_backtracking_policy:
        restriction, solve_time = cheap_a_star_policy(graph, vertex, state, target_state)
    else:
        raise Exception("not selected policy")
        
    return restriction, solve_time



# ------------------------------------------------------------------------
# functions you'd probably never need to use

def get_k_step_lookahead_cost_to_go(
    graph: PolynomialDualGCS,
    lookahead:int,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> float:
    graph.options.policy_lookahead_horizon = lookahead
    options = graph.options
    options.vertify_options_validity()

    if target_state is None:
        if options.potentials_are_not_a_function_of_target_state:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()

    node = RestrictionSolution([vertex], [initial_state])
    que = get_k_step_optimal_paths(graph, node)[0]
    return que.get()[0]


# ------------------------------------------------------------------------

# helper functions for pasth postprocessing

def make_a_list_of_shortcuts(numbers:T.List[int], K:int, index:int=0):
    assert 0 <= index and index < len(numbers)
    res = []
    for i in range(0,K+1):
        if numbers[index] - i >= 1:
            if index == len(numbers)-1:        
                res.append( numbers[:index] + [numbers[index] - i] )
            else:
                res = res + make_a_list_of_shortcuts( numbers[:index] + [numbers[index] - i] + numbers[index+1:], K, index+1 )
        else:
            break
    return res

def make_list_of_shortcuts_for_index(numbers:T.List[int], K:int, index:int):
    assert 0 <= index and index < len(numbers)
    res = []
    for i in range(0,K+1):
        if numbers[index] - i >= 1:
            res.append( numbers[:index] + [numbers[index] - i] + numbers[index+1:] )
        else:
            break
    return res

def get_name_repeats(vertex_names):
    unique_vertex_names = []
    repeats = []
    i = 0
    while i < len(vertex_names):
        unique_vertex_names.append(vertex_names[i])
        r = 1
        while i+1 < len(vertex_names) and vertex_names[i+1] == vertex_names[i]:
            r += 1
            i += 1
        repeats.append(r)
        i+=1
    return unique_vertex_names, repeats

def repeats_to_vertex_names(vertices, repeats):
    res = []
    for i in range(len(repeats)):
        res += [vertices[i]] * repeats[i]
    return res

def get_warmstart_for_a_shortcut(restriction:RestrictionSolution, 
                                 og_repeats:T.List[int], 
                                 new_repeats:T.List[int], 
                                 unique_vertices:T.List[DualVertex]):
    new_vertex_path = repeats_to_vertex_names(unique_vertices, new_repeats)
    new_trajectory = []
    new_edge_var_trajectory = []
    index = 0
    for i, og_repeat in enumerate(og_repeats):
        new_repeat = new_repeats[i]
        j = 0
        while j < new_repeat:
            if index < len(restriction.trajectory):
                new_trajectory.append(restriction.trajectory[index])
            if index < len(restriction.edge_variable_trajectory):
                new_edge_var_trajectory.append(restriction.edge_variable_trajectory[index])
            index += 1
            j+=1
        index += (og_repeat-new_repeat)
    assert len(new_trajectory) == len(new_vertex_path)
    assert len(new_trajectory) == len(new_edge_var_trajectory) + 1
    return RestrictionSolution(new_vertex_path, new_trajectory, new_edge_var_trajectory)
