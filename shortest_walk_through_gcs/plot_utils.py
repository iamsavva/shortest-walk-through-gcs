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
    VPolytope,
    Hyperellipsoid
)

from scipy.special import comb

import plotly.graph_objects as go  # pylint: disable=import-error

from shortest_walk_through_gcs.gcs_dual import PolynomialDualGCS, DualEdge, DualVertex




# ------------------------------------------------------------------
# bezier plot utils


def get_bezier_point_combination(i, N, t):
    val = comb(N, i) * t**i * (1.0 - t) ** (N - i)
    return val


tt = np.linspace(0, 1, 100)


def MakeBezier(t: npt.NDArray, X: npt.NDArray):
    """
    Evaluates a Bezier curve for the points in X.

    Args:
        t (npt.NDArray): npt.ND: number (or list of numbers) in [0,1] where to evaluate the Bezier curve
        X (npt.NDArray): list of control points

    Returns:
        npt.NDArray: set of 2D points along the Bezier curve
    """
    N, d = np.shape(X)  # num points, dim of each point
    assert d == 2, "control points need to be dimension 2"
    xx = np.zeros((len(t), d))
    # evaluate bezier per row
    for i in range(N):
        xx += np.outer(get_bezier_point_combination(i, N - 1, t), X[i])
    return xx


def plot_linear_segments(
    fig: go.Figure,
    linear_segments: T.List[npt.NDArray],
    color="purple",
    name=None,
    linewidth=3,
    marker_size = 3,
    maker_outline_width = 8,
    plot_start_point = True,
    dotted=True,
    plot_start_target_only=False
):
    showlegend = False
    line_name = ""
    if name is not None:
        line_name = name
        showlegend = True

    full_path = None
    first = True

    if plot_start_target_only:
        fig.add_trace(
            go.Scatter(
                x=[linear_segments[0][0], linear_segments[-1][0]],
                y=[linear_segments[0][1], linear_segments[-1][1]],
                mode="markers",
                marker=dict(
                    color='white',        # Set the fill color of the markers
                    size=marker_size,             # Set the size of the markers
                    line=dict(
                        color=color,     # Set the outline color of the markers
                        width=maker_outline_width          # Set the width of the marker outline
                    )
                ),
                showlegend=False,
            )
        )
        return

    line_dict = dict(width=linewidth)
    if dotted: 
        line_dict = dict(width=linewidth, dash='dot')

    # plot bezier curves
    fig.add_trace(
        go.Scatter(
            x=linear_segments[:, 0],
            y=linear_segments[:, 1],
            marker_color=color,
            line=line_dict,
            mode="lines+markers",
            name=line_name,
            marker=dict(
                    color='white',        # Set the fill color of the markers
                    size=marker_size,             # Set the size of the markers
                    line=dict(
                        color=color,     # Set the outline color of the markers
                        width=maker_outline_width          # Set the width of the marker outline
                    )
                ),
            showlegend=first and line_name != "",
        )
    )


def plot_bezier(
    fig: go.Figure,
    bezier_curves: T.List[T.List[npt.NDArray]],
    bezier_color="purple",
    control_point_color="red",
    name=None,
    linewidth=3,
    marker_size = 3,
    plot_start_point = True,
    dotted=True,
    plot_start_target_only=False,
    plot_start_target = True,
    plot_bezier_start_end = True,
    plot_control_points = True,
    width_of_marker_outline = 8,
):
    showlegend = False
    line_name = ""
    if name is not None:
        line_name = name
        showlegend = True

    full_path = None
    first = True

    if plot_start_target_only or plot_start_target:
        fig.add_trace(
            go.Scatter(
                x=[bezier_curves[0][0][0], bezier_curves[-1][-1][0]],
                y=[bezier_curves[0][0][1], bezier_curves[-1][-1][1]],
                mode="markers",
                marker=dict(
                    color='white',        # Set the fill color of the markers
                    size=marker_size,             # Set the size of the markers
                    line=dict(
                        color=control_point_color,     # Set the outline color of the markers
                        width=width_of_marker_outline          # Set the width of the marker outline
                    )
                ),
                showlegend=False,
            )
        )
        if plot_start_target_only:
            return



    for curve_index, bezier_curve in enumerate(bezier_curves):
        X = np.array(bezier_curve)
        tt = np.linspace(0, 1, 100)
        xx = MakeBezier(tt, X)
        if full_path is None:
            full_path = xx
        else:
            full_path = np.vstack((full_path, xx))

        # plot bezier curves
        if dotted:
            fig.add_trace(
                go.Scatter(
                    x=full_path[:, 0],
                    y=full_path[:, 1],
                    marker_color=bezier_color,
                    line=dict(width=linewidth, dash='dot'),
                    mode="lines",
                    name=line_name,
                    showlegend=first and line_name != "",
                )
            )
        else: 
            fig.add_trace(
                go.Scatter(
                    x=full_path[:, 0],
                    y=full_path[:, 1],
                    marker_color=bezier_color,
                    line=dict(width=linewidth),
                    mode="lines",
                    name=line_name,
                    showlegend=first and line_name != "",
                )
            )
        first = False

    for curve_index, bezier_curve in enumerate(bezier_curves):
        X = np.array(bezier_curve)
        tt = np.linspace(0, 1, 100)
        xx = MakeBezier(tt, X)
        if full_path is None:
            full_path = xx
        else:
            full_path = np.vstack((full_path, xx))
        # plot contorl points
        if control_point_color is not None and plot_bezier_start_end:
            fig.add_trace(
                go.Scatter(
                    x=[X[0, 0], X[-1, 0]],
                    y=[X[0, 1], X[-1, 1]],
                    # marker_color=control_point_color,
                    marker_symbol="circle",
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color='white',        # Set the fill color of the markers
                        size=marker_size,             # Set the size of the markers
                        line=dict(
                            color=control_point_color,     # Set the outline color of the markers
                            width=width_of_marker_outline          # Set the width of the marker outline
                        )
                    ),
                )
            )
        if control_point_color is not None and plot_control_points:
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0],
                    y=X[:, 1],
                    # marker_color=control_point_color,
                    marker_symbol="circle",
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color='white',        # Set the fill color of the markers
                        size=marker_size,             # Set the size of the markers
                        line=dict(
                            color=control_point_color,     # Set the outline color of the markers
                            width=width_of_marker_outline          # Set the width of the marker outline
                        )
                    ),
                )
            )

        if plot_start_point:
            if curve_index == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[xx[0, 0]],
                        y=[xx[0, 1]],
                        # marker_color=bezier_color,
                        mode="markers",
                        marker=dict(
                        color='white',        # Set the fill color of the markers
                        size=marker_size,             # Set the size of the markers
                        line=dict(
                            color=control_point_color,     # Set the outline color of the markers
                            width=width_of_marker_outline          # Set the width of the marker outline
                        )
                    ),
                        showlegend=False,
                    )
                )

def get_ellipse(mu, sigma):
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    mu = np.array(mu).reshape((2,1))
    theta = np.linspace(0, 2*np.pi, 1000)
    ellipse = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)] + mu
    xs=ellipse[0,:]
    ys=ellipse[1,:]
    return xs,ys

def plot_a_2d_graph(vertices:T.List[DualVertex], width = 800, fill_color = "mintcream"):
    fig = go.Figure()

    def add_trace(convex_set:ConvexSet):
        if isinstance(convex_set, Hyperrectangle):
            lb,ub = convex_set.lb(), convex_set.ub()
            xs = [lb[0], lb[0], ub[0], ub[0], lb[0]]
            ys = [lb[1], ub[1], ub[1], lb[1], lb[1]]
        elif isinstance(convex_set, HPolyhedron):
            vertices = get_clockwise_vertices(VPolytope(convex_set))
            xs = np.append(vertices[0,:], vertices[0,0])
            ys = np.append(vertices[1,:], vertices[1,0])
        elif isinstance(convex_set, Hyperellipsoid):
            mu = convex_set.center()
            sigma = np.linalg.inv(convex_set.A().T.dot(convex_set.A()))
            xs,ys = get_ellipse(mu,sigma)
        elif isinstance(convex_set, Point):
            xs = [convex_set.x()[0]]
            ys = [convex_set.x()[1]]
        else:
            raise Exception("what the bloody hell is that set")

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                line=dict(color="black"),
                fillcolor=fill_color,
                fill="tozeroy",
                showlegend=False,
            )
        )

    for v in vertices:
        add_trace(v.convex_set)
        if isinstance(v.convex_set, Hyperrectangle):
            center = (v.convex_set.lb() + v.convex_set.ub()) / 2
        elif isinstance(v.convex_set, HPolyhedron):
            vertices = get_clockwise_vertices(VPolytope(v.convex_set))
            center = np.mean(vertices, axis=1)
        elif isinstance(v.convex_set, Hyperellipsoid):
            center = v.convex_set.center()

        fig.add_trace(
            go.Scatter(
                x=[center[0]],
                y=[center[1]+0.5],
                mode="text",
                text=[v.name],
                showlegend=False,
            )
        )

    fig.update_layout(height=width, width=width, title_text="Graph view")
    fig.update_layout(
        yaxis=dict(scaleanchor="x"),  # set y-axis to have the same scaling as x-axis
        yaxis2=dict(
            scaleanchor="x", overlaying="y", side="right"
        ),  # set y-axis2 to have the same scaling as x-axis
    )

    return fig


def get_clockwise_vertices(vpoly:VPolytope):
    vertices = list(vpoly.vertices().T)
    c = np.mean(vpoly.vertices(),axis=1) # compute center
    vertices.sort(key = lambda p: np.arctan2( (p-c)[1], (p-c)[0] ) )
    return np.array(vertices).T

