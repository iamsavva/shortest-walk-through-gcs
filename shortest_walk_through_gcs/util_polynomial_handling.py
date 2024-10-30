import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
)
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    Hyperellipsoid,
)

from pydrake.symbolic import (  # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
    Monomial,
)
from pydrake.math import ( # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)  

from shortest_walk_through_gcs.program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

from shortest_walk_through_gcs.util_moments import make_moment_matrix

def define_quadratic_polynomial(
    prog: MathematicalProgram,
    x: npt.NDArray,
    potential_type: str,
    specific_J_matrix: npt.NDArray = None,
) -> T.Tuple[npt.NDArray, Expression]:
    # specific potential provided
    state_dim = len(x)

    if specific_J_matrix is not None:
        assert specific_J_matrix.shape == (state_dim + 1, state_dim + 1)
        J_matrix = specific_J_matrix
    else:
        # potential is a free polynomial
        # TODO: change order to match Russ's notation

        if potential_type == FREE_POLY:
            J_matrix = prog.NewSymmetricContinuousVariables(state_dim + 1)

        elif potential_type == PSD_POLY:
            J_matrix = prog.NewSymmetricContinuousVariables(state_dim + 1)
            prog.AddPositiveSemidefiniteConstraint(J_matrix)

        elif potential_type == CONVEX_POLY:
            # i don't actually care about the whole thing being PSD;
            # only about the x component being convex
            J_matrix = prog.NewSymmetricContinuousVariables(state_dim + 1)
            prog.AddPositiveSemidefiniteConstraint(J_matrix[1:, 1:])

            # NOTE: Russ uses different notation for PSD matrix. the following won't work
            # NOTE: i am 1 x^T; x X. Russ is X x; x^T 1
            # self.potential, self.J_matrix = prog.NewSosPolynomial( self.vars, 2 )
        else:
            raise NotImplementedError("potential type not supported")

    x_and_1 = np.hstack(([1], x))
    potential = x_and_1.dot(J_matrix).dot(x_and_1)

    return J_matrix, potential

def make_potential(indet_list: npt.NDArray, potential_type:str, poly_deg:int, prog: MathematicalProgram) -> T.Tuple[Expression, npt.NDArray]:
    state_dim = len(indet_list)
    vars_from_indet = Variables(indet_list)
    # quadratic polynomial. special case due to convex functions and special quadratic implementations
    if poly_deg == 0:
        a = np.zeros(state_dim)
        b = prog.NewContinuousVariables(1)[0]
        J_matrix = make_moment_matrix(b, a, np.zeros((state_dim, state_dim)))
        potential = Expression(b)
        if potential_type == PSD_POLY:
            prog.AddLinearConstraint(b >= 0)
            
    elif poly_deg == 1:
        a = prog.NewContinuousVariables(state_dim)
        b = prog.NewContinuousVariables(1)[0]
        J_matrix = make_moment_matrix(b, a, np.zeros((state_dim, state_dim)))
        potential = 2 * a.dot(indet_list) + b
    elif poly_deg == 2:
        J_matrix, potential = define_quadratic_polynomial(prog, indet_list, potential_type)
    else:
        J_matrix = None
        # free polynomial
        if potential_type == FREE_POLY:
            potential = prog.NewFreePolynomial(
                vars_from_indet, poly_deg
            ).ToExpression()
        # PSD polynomial
        elif potential_type == PSD_POLY:
            assert (
                poly_deg % 2 == 0
            ), "can't make a PSD potential of uneven degree"
            # potential is PSD polynomial
            potential = prog.NewSosPolynomial(vars_from_indet, poly_deg)[0].ToExpression()
        else:
            raise NotImplementedError("potential type not supported")
    return potential, J_matrix

def define_sos_constraint_over_polyhedron_multivar_new(
    prog: MathematicalProgram,
    unique_vars: npt.NDArray,
    linear_inequalities: T.List[Expression],
    quadratic_inequalities: T.List[Expression],
    equality_constraints: T.List[Expression],
    subsitution_dictionary: T.Dict[Variable, Expression],
    function: Expression,
    options: ProgramOptions,
) -> None:
    all_variables = Variables(unique_vars)
    def make_multipliers(degree, dimension, psd):
        if degree == 0:
            lambdas = prog.NewContinuousVariables(dimension)
            if psd:
                prog.AddLinearConstraint(ge(lambdas, 0))
        else:
            if psd:
                lambdas = [
                    prog.NewSosPolynomial(all_variables, degree)[0].ToExpression()
                    for _ in range(dimension)
                ]
            else:
                lambdas = [
                    prog.NewFreePolynomial(all_variables, degree).ToExpression()
                    for _ in range(dimension)
                ]
        return lambdas

    linear_inequalities = np.hstack(linear_inequalities) if len(linear_inequalities) > 0 else np.array(linear_inequalities)
    quadratic_inequalities = np.hstack(quadratic_inequalities) if len(quadratic_inequalities) > 0 else np.array(quadratic_inequalities)
    equality_constraints = np.hstack(equality_constraints) if len(equality_constraints) > 0 else np.array(equality_constraints)

    for i in range(len(linear_inequalities)):
        linear_inequalities[i] = linear_inequalities[i].Substitute(subsitution_dictionary)

    for i in range(len(quadratic_inequalities)):
        quadratic_inequalities[i] = quadratic_inequalities[i].Substitute(subsitution_dictionary)
    
    for i in range(len(equality_constraints)):
        equality_constraints[i] = equality_constraints[i].Substitute(subsitution_dictionary)


    s_procedure = Expression(0)

    # deg 0
    lambda_0 = prog.NewContinuousVariables(1)[0]
    prog.AddLinearConstraint(lambda_0 >= 0)
    s_procedure += lambda_0

    # deg 1
    deg = options.s_procedure_multiplier_degree_for_linear_inequalities
    if len(linear_inequalities) > 0:
        lambda_1 = make_multipliers(deg, len(linear_inequalities), True)
        s_procedure += linear_inequalities.dot(lambda_1)
        # deg 1 products
        if options.s_procedure_take_product_of_linear_constraints:
            for i in range(len(linear_inequalities)-1):
                lambda_1_prod = make_multipliers(deg, len(linear_inequalities) - (i+1), True)
                s_procedure += linear_inequalities[i+1:].dot(lambda_1_prod) * linear_inequalities[i]

    # deg 2
    if len(quadratic_inequalities) > 0:
        lambda_2 = make_multipliers(deg, len(quadratic_inequalities), True)
        s_procedure += quadratic_inequalities.dot(lambda_2)

    # equality constraints
    if len(equality_constraints) > 0:
        lambda_eq = make_multipliers(deg, len(equality_constraints), False)
        s_procedure += equality_constraints.dot(lambda_eq)

    
    expr = function.Substitute(subsitution_dictionary) - s_procedure

    prog.AddSosConstraint(expr, monomial_basis=[Monomial(mon) for mon in unique_vars])

def get_set_membership_inequalities(x:npt.NDArray, convex_set:ConvexSet):
    linear_inequalities = []
    quadratic_inequalities = []
    x_and_1 = np.hstack(([1], x))

    if isinstance(convex_set, Hyperrectangle) or isinstance(convex_set, HPolyhedron):
        if isinstance(convex_set, Hyperrectangle):
            hpoly = convex_set.MakeHPolyhedron()
        else:
            hpoly = convex_set
        # inequalities of the form b[i] - a.T x = g_i(x) >= 0
        A, b = hpoly.A(), hpoly.b()
        B = np.hstack((b.reshape((len(b), 1)), -A))
        linear_inequalities = B.dot(x_and_1)

    elif isinstance(convex_set, Hyperellipsoid):
        mu = convex_set.center()
        A = convex_set.A()
        c11 = 1 - mu.T.dot(A.T).dot(A).dot(mu)
        c12 = mu.T.dot(A.T).dot(A)
        c21 = A.T.dot(A).dot(mu).reshape((len(mu), 1))
        c22 = -A.T.dot(A)
        E = np.vstack((np.hstack((c11,c12)), np.hstack((c21,c22))))
        quadratic_inequalities = [np.sum( E * (np.outer(x_and_1, x_and_1)) )]

    return linear_inequalities, quadratic_inequalities

def get_set_intersection_inequalities(x:npt.NDArray, convex_set:ConvexSet, another_set:ConvexSet):
    if isinstance(convex_set, Point) or isinstance(another_set, Point):
        return [],[]

    if isinstance(convex_set, Hyperellipsoid) or isinstance(another_set, Hyperellipsoid):
        l1, q1 = get_set_membership_inequalities(x, convex_set)
        l2, q2 = get_set_membership_inequalities(x, another_set)
        return l1+l2, q1+q2
    else:
        if isinstance(convex_set, Hyperrectangle):
            hpoly1 = convex_set.MakeHPolyhedron()
        else:
            hpoly1 = convex_set

        if isinstance(another_set, Hyperrectangle):
            hpoly2 = another_set.MakeHPolyhedron()
        else:
            hpoly2 = another_set

        hpoly = hpoly1.Intersection(hpoly2, check_for_redundancy=True)
        return get_set_membership_inequalities(x, hpoly)