import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.all import MosekSolver, ClarabelSolver  # pylint: disable=import-error, no-name-in-module, unused-import

from shortest_walk_through_gcs.util import timeit, INFO, YAY, ERROR, WARN  # pylint: disable=unused-import

# polynomial types
FREE_POLY = "free_poly" # free polynomial. more expressive, but makes the policy rollouts non-convex
PSD_POLY = "psd_poly" # PSD polynomial: non-negative and convex
CONVEX_POLY = "convex_poly" # convex polynomial, but not necessarily non-negative. only relevant for degree 2 polynomials


class ProgramOptions:
    def __init__(self):
        # -----------------------------------------------------------------------------------
        # settings pertaining to the cost-to-go synthesis program
        # -----------------------------------------------------------------------------------
        # i refer to potentials and cost-to-go lower bounds interchangeably


        # ----------------------------------
        # lower bound potentials settings
        # offline stage produces polynomial cost-to-go lower bounds. these settings govern the complexity of these lower bounds
        self.potential_polynomial_degree = 2 # degree of the polynomial lower bound. at least 0. in practice, 0,1,2.
        self.potential_type = PSD_POLY # see polynomial types above

        # ----------------------------------
        # S-procedure
        self.s_procedure_multiplier_degree_for_linear_inequalities = 0 # must be even. higher = more expressive cost-to-go = larger program
        self.s_procedure_take_product_of_linear_constraints = True # True = more expressive cost-to-go = larger program

        # ------------------
        # solver that's used for cost-to-go function synthesis
        # if you don't have a license -- use ClarabelSolver
        self.cost_to_go_synthesis_solver = MosekSolver

        # ----------------------------------
        # additional settings if MOSEK is used for cost-to-go function synthesis
        self.cost_to_go_synthesis_use_robust_mosek_parameters = True
        self.cost_to_go_synthesis_MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-8
        self.cost_to_go_synthesis_MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-8
        self.cost_to_go_synthesis_MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-8

        # ----------------------------------
        # special important settings

        # if True, then don't use penalties. solving Shortest Walk Problem in GCS
        # if False, then add penalties. solving Shortest Path Problem in GCS
        self.solve_shortest_walk_not_path = False

        # is solving the Shortest Path Problem, then set the flow vilation polynomial degree
        self.flow_violation_polynomial_degree = 0 # even, 0 or 2

        # consider transition (x_v, x_w) over the edge e=(v,w)
        # if the following flag is True, then x_w \in X_v \cap X_w
        # otherwise, x_w \ in X_w.
        # TODO: this should really be a per-edge flag.
        self.right_edge_point_must_be_inside_intersection_of_left_and_right_sets = False

        # whether to synthesize a cost-to-go lower bounds that are a function of the target state
        # if False, then must go from x_s to specific x_t, 
        # and lower bounds are a fucntion of target state J_v(x_v,x_t).
        # if True, then must from from x_s to any x_t \in X_t
        # and lower bounds area function are just J_v(x_v)
        self.potentials_are_not_a_function_of_target_state = False




        # -----------------------------------------------------------------------------------
        # settings pertaining to the incremental search
        # -----------------------------------------------------------------------------------

        # solver selection for solving lookahead programs during incremental search
        self.policy_solver = MosekSolver

        # ---------------------------------------------
        # specify the policy

        # must select one or the other
        self.use_greedy_with_backtracking_policy = False
        self.use_a_star_with_limited_backtracking_policy = False

        # ---------------------------------------------
        # lookahead horizon for the policy and 
        # at each iteration, we pick a k-step optimal action sequence, and execute just the first step.
        self.policy_lookahead_horizon = 1
        # if True, then at each iteration we re-optimize the path so far and the next K-step sequence simultaniously
        self.policy_optimize_path_so_far_and_K_step = False

        # terminate if the solution was not acquired after this many iterations
        self.iteration_termination_limit = 1000

        # limited backtracking in A-star
        # if you currently expanded an n-step path, do not consider paths of length (n-limit)
        self.a_star_backtracking_limit = 2        

        # do not select a specfic k-step lookahead path as optimal more than this many times
        # this helps avoid expanding the same aaa vertex sequence a thousand times and not making any forward progress
        self.subpath_expansion_limit = 10

        # a hack:
        # if you know that your program is such that a walk would only ever revisit a vertex in sequence,
        # i.e., aaabbbbccc, but never  abababc
        # then set the next flag to True. this would reduce the search space.
        self.do_not_return_to_previously_visited_vertices = False


        # -----------------------------------------------------------------------------------
        # settings pertaining to postprocessing of the walk
        # -----------------------------------------------------------------------------------
        self.postprocess_by_solving_restriction_on_mode_sequence = True

        self.postprocess_via_shortcutting = False
        self.max_num_shortcut_steps = 1

        self.postprocess_shortcutting_long_sequences = False
        self.long_sequence_num = 4
        



        # ---------------------------------------------
        # mosek-specific options for policy rollouts
        self.policy_MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-6
        self.policy_MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-6
        self.policy_MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-6
        self.MSK_DPAR_INTPNT_TOL_INFEAS = 1e-6
        self.MSK_IPAR_PRESOLVE_USE = 1 # 0: use, 1: don't
        self.MSK_IPAR_INTPNT_SOLVE_FORM = 2 # 0: pick, 1: primal, 2: dual
        self.MSK_IPAR_OPTIMIZER = None # free, intpnt, conic, primal_simplex, dual_simplex, free_simplex, mixed_int 

        # ---------------------------------------------
        # snopt-specific options for poluicy rollouts
        self.policy_snopt_minor_iterations_limit = 500
        self.policy_snopt_major_iterations_limit = 1000
        self.policy_snopt_minor_feasibility_tolerance = 1e-6
        self.policy_snopt_major_feasibility_tolerance = 1e-6
        self.policy_snopt_major_optimality_tolerance = 1e-6
        # whether to use warmstarting. only really relevant for SNOPT, as all other programs are solved with IPM solvers
        self.policy_use_warmstarting = False
        

        # verbosing
        self.verbose_solve_times = False
        self.policy_verbose_choices = False
        self.policy_verbose_number_of_restrictions_solves = False
        self.verbose_restriction_improvement = False
        

        # ----------------------------------
        # GCS policy related settings.
        # as of now, only used for computing optimal solutions.
        self.gcs_policy_solver = MosekSolver
        self.gcs_policy_use_convex_relaxation = True
        self.gcs_policy_max_rounding_trials = 300
        self.gcs_policy_max_rounded_paths = 300
        self.gcs_policy_use_preprocessing = True


        # ---------------------------------------------
        # more subtle or hacky options; just do not change them
        # ---------------------------------------------

        self.so_far_and_k_step_ratio = 1
        self.so_far_and_k_step_initials = 5
        
        self.relax_target_condition_during_rollout = False

        self.policy_use_zero_heuristic = False 

        self.check_cost_to_go_at_point = False # set to False, faster rollout prog generation

        # TODO: FIX THIS. this is a temporary hack.
        self.use_skill_compoisition_constraint_add = False


        # solve time reporting specific nit
        self.use_parallelized_solve_time_reporting = True
        self.num_simulated_cores_for_parallelized_solve_time_reporting = 16

        

        

    def vertify_options_validity(self):
        assert self.policy_lookahead_horizon >= 1, "lookahead must be positive"
        assert self.potential_type in (
            FREE_POLY,
            PSD_POLY,
            CONVEX_POLY,
        ), "undefined potentia type"
        policy_options = np.array([self.use_greedy_with_backtracking_policy, self.use_a_star_with_limited_backtracking_policy])
        assert not np.sum(policy_options) < 1, "must select policy lookahead option"
        assert not np.sum(policy_options) > 1, "selected multiple policy lookahead options"




