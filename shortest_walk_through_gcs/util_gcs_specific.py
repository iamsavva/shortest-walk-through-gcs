import typing as T

import numpy as np
import numpy.typing as npt  # pylint: disable=unused-import
import pydot  # pylint: disable=import-error

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module
    MathematicalProgramResult,
)

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    GraphOfConvexSets,
)

from shortest_walk_through_gcs.util import (
    timeit,
    INFO,
    YAY,
    ERROR,
    WARN,
    diditwork,
)  # pylint: disable=unused-import




def get_vertex_name(t: int, set_name: str) -> str:
    return "T" + str(t) + " " + set_name


def get_mode_from_name(vertex_name: str) -> str:
    return vertex_name.split(" ")[1]

def get_edge_name(left_name: str, right_name: str) -> str:
    return left_name + " " + right_name

def plot_a_gcs(gcs, solution=None, graph_name="temp"):
    if solution is None:
        graphviz = gcs.GetGraphvizString()
    else:
        graphviz = gcs.GetGraphvizString(solution, True, precision=2)

    data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
    data.write_png(graph_name + ".png")
    data.write_svg(graph_name + ".svg")
