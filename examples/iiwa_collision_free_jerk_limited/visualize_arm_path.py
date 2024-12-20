from arm_visualization import ArmComponents, arm_components_loader
from pydrake.perception import PointCloud

from pydrake.geometry import Rgba

import os

from pydrake.all import Trajectory, CompositeTrajectory


from pydrake.all import ( # pylint: disable=import-error, no-name-in-module, unused-import
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseDynamicsController,
    LogVectorOutput,
    MeshcatVisualizer,
    MultibodyPlant,
    PiecewisePolynomial,
    StartMeshcat,
    TrajectorySource,
)

import pickle

from pydrake.all import ( # pylint: disable=import-error, no-name-in-module, unused-import
    BsplineBasis,
    BsplineTrajectory,
    Context,
    Diagram,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Trajectory,
    TrajectorySource,
    VectorLogSink,
    Simulator,
    Parser,
)


from shortest_walk_through_gcs.util import diditwork, ERROR, WARN, INFO, timeit

import numpy as np


import typing as T
import numpy.typing as npt

from dataclasses import dataclass

import os

from pydrake.trajectories import Trajectory, BezierCurve, CompositeTrajectory, PathParameterizedTrajectory # pylint: disable=import-error, no-name-in-module, unused-import
from pydrake.multibody.optimization import Toppra # pylint: disable=import-error, no-name-in-module, unused-import
from pydrake.all import RandomGenerator, ConvexSet # pylint: disable=import-error, no-name-in-module, unused-import

from manipulation.utils import ConfigureParser



import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    VPolytope,
    LoadIrisRegionsYamlFile,
    SaveIrisRegionsYamlFile
)


from pydrake.symbolic import ( # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)  

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    MosekSolverDetails,
    SolverOptions,
    CommonSolverOption,
    IpoptSolver,
    SnoptSolver,
    GurobiSolver,
    OsqpSolver,
    ClarabelSolver,
)
import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots


import plotly
import plotly.graph_objs as go
from IPython.display import display, HTML

plotly.offline.init_notebook_mode()
display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
))

import logging


from tqdm import tqdm
from pydrake.all import RandomGenerator, PathParameterizedTrajectory
from pydrake.math import RigidTransform, RollPitchYaw



from arm_visualization import visualize_arm_at_state, visualize_arm_at_samples_from_set, save_the_visualization, arm_components_loader, ArmComponents, create_arm, visualize_arm_at_state, make_path_paramed_traj_from_list_of_bezier_curves, Simulator
from iiwa_helpers import triple_integrator_postprocessing


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


from arm_visualization import ArmComponents, arm_components_loader
from pydrake.perception import PointCloud

from pydrake.geometry import Rgba

import os


from matplotlib.pyplot import get_cmap

from manipulation.utils import ConfigureParser
from pydrake.all import Parser
import numpy as np

from pydrake.all import ( # pylint: disable=import-error, no-name-in-module, unused-import
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseDynamicsController,
    LogVectorOutput,
    MeshcatVisualizer,
    MultibodyPlant,
    PiecewisePolynomial,
    StartMeshcat,
    TrajectorySource,
)

from pydrake.all import ( # pylint: disable=import-error, no-name-in-module, unused-import
    BsplineBasis,
    BsplineTrajectory,
    Context,
    Diagram,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Trajectory,
    TrajectorySource,
    VectorLogSink,
    Simulator,
    Parser,
)


import numpy as np


import typing as T
import numpy.typing as npt

from dataclasses import dataclass

import os

from pydrake.trajectories import Trajectory, BezierCurve, CompositeTrajectory, PathParameterizedTrajectory # pylint: disable=import-error, no-name-in-module, unused-import
from pydrake.multibody.optimization import Toppra # pylint: disable=import-error, no-name-in-module, unused-import
from pydrake.all import RandomGenerator, ConvexSet # pylint: disable=import-error, no-name-in-module, unused-import

from manipulation.utils import ConfigureParser



def ForwardKinematics(q_list):
    """Returns the end-effector pose for the given joint angles.

    The end-effector is the body of the wsg gripper.

    Args:
        q_list: List of joint angles.

    Returns:
        List of end-effector poses.
    """
    arm_components = arm_components_loader(use_meshcat=False)

    plant = arm_components.plant
    plant_context = plant.GetMyContextFromRoot(arm_components.diagram_context)

    X_list = []
    for q in q_list:
        plant.SetPositions(plant_context, q)
        X_list.append(
            plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body")))

    return X_list

def get_colormap_rgba(value, colormap_name='viridis'):    
    colormap = get_cmap(colormap_name)
    rgba = colormap(value)  # Get the RGBA value
    return rgba



def get_normalized_constraint_riding(v_traj, a_traj, add_one_at_end=True):
    arm_components = arm_components_loader(use_meshcat=False)
    vel_ub = arm_components.plant.GetVelocityUpperLimits()
    acc_ub = arm_components.plant.GetAccelerationUpperLimits()

    vel_cons = np.clip(np.max(np.abs(v_traj)/vel_ub, axis=1), None, 1)

    acc_cons = np.max(np.abs(a_traj)/acc_ub, axis=1)
    if add_one_at_end:
        acc_cons=np.append(acc_cons,acc_cons[-1])
    return vel_cons, acc_cons


def get_normalized_constraint_riding_with_jerk(v_traj, a_traj, j_traj):
    arm_components = arm_components_loader(use_meshcat=False)
    vel_ub = arm_components.plant.GetVelocityUpperLimits()
    acc_ub = arm_components.plant.GetAccelerationUpperLimits()
    jerk_ub = acc_ub*(acc_ub/vel_ub)

    vel_cons = np.max(np.abs(v_traj)/vel_ub, axis=1)
    jerk_cons = np.max(np.abs(j_traj)/jerk_ub, axis=1)
    acc_cons = np.max(np.abs(a_traj)/acc_ub, axis=1)
    return vel_cons, acc_cons, jerk_cons

def visualize_trajectory(pos_traj, vel_traj, time_traj):
    # arm_components = arm_components_loader()
    use_meshcat = True
    arm_file_path="./models/iiwa14_david_cheap_bigger3.dmd.yaml"
    num_joints = 7
    time_step = 0.0

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.package_map().AddPackageXml(filename=os.path.abspath("./models/package.xml"))

    # Add arm
    parser.AddModels(arm_file_path)
    try:
        arm = plant.GetModelInstanceByName("iiwa")
    except:
        arm = plant.GetModelInstanceByName("arm")


    plant.Finalize()

    placeholder_trajectory = PiecewisePolynomial(np.zeros((num_joints, 1)))
    trajectory_source = builder.AddSystem(
        TrajectorySource(placeholder_trajectory, output_derivative_order=1)
    )


    # Meshcat
    if use_meshcat:
        meshcat = StartMeshcat()
        if num_joints < 3:
            meshcat.Set2dRenderMode()
        meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat
        )
    else:
        meshcat = None
        meshcat_visualizer = None

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    # plant_context = plant.GetMyContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)

    arm_components = ArmComponents(
        num_joints=7,
        diagram=diagram,
        plant=plant,
        trajectory_source=trajectory_source,
        meshcat=meshcat,
        meshcat_visualizer=meshcat_visualizer,
        diagram_context=diagram_context
    )

    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    plant_context = arm_components.plant.GetMyContextFromRoot(context)
    arm_components.meshcat_visualizer.StartRecording()

    arm_components.meshcat.SetCameraPose([-1,0.4,1],[0,0,0.3])


    for q, q_dot, t in zip(pos_traj, vel_traj, time_traj):
        arm_components.plant.SetPositions(plant_context, q)
        arm_components.plant.SetVelocities(plant_context, q_dot) 
        simulator.AdvanceTo(t)


    # replay trajectories
    arm_components.meshcat_visualizer.StopRecording()
    arm_components.meshcat_visualizer.PublishRecording()



def reparameterize_with_toppra(
    trajectory: Trajectory,
    num_grid_points: int = 1000,
    add_acc_limits=True,
) -> PathParameterizedTrajectory:
    arm_components=arm_components_loader(use_meshcat=False)
    plant = arm_components.plant
    toppra = Toppra(
        path=trajectory,
        plant=plant,
        gridpoints=np.linspace(
            trajectory.start_time(), trajectory.end_time(), num_grid_points
        ),
    )
    velocity_limits = plant.GetVelocityUpperLimits()
    acceleration_limits=plant.GetAccelerationUpperLimits()

    toppra.AddJointVelocityLimit(-velocity_limits, velocity_limits)
    if add_acc_limits:
        toppra.AddJointAccelerationLimit(-acceleration_limits, acceleration_limits)
    timer = timeit()
    time_trajectory = toppra.SolvePathParameterization()
    toppra_solve_tome = timer.dt("toppra solve", False)
    return PathParameterizedTrajectory(trajectory, time_trajectory), toppra_solve_tome


def get_composite_trajectory_values(traj: CompositeTrajectory, N = 200):
    time_traj = np.linspace(traj.start_time(), traj.end_time(), N)
    pos_traj = traj.vector_values(time_traj).T
    vel_traj = traj.MakeDerivative(1).vector_values(time_traj).T
    acc_traj = traj.MakeDerivative(2).vector_values(time_traj).T
    jerk_traj = traj.MakeDerivative(3).vector_values(time_traj).T
    return time_traj, pos_traj, vel_traj, acc_traj, jerk_traj


def get_my_trajectory_values(time_trajectory, position_trajectory, velocity_trajectory, acceleration_trajectory, N=100):

    small_time_traj = np.hstack([np.linspace(time_trajectory[i], time_trajectory[i+1], N)[:-1] for i in range(len(time_trajectory)-1)] + [time_trajectory[-1]])
    small_pos_traj = np.empty((len(small_time_traj), 7))
    small_vel_traj = np.empty((len(small_time_traj), 7))
    small_acc_traj = np.empty((len(small_time_traj)-1, 7))
    for i in range(len(time_trajectory)-1):
        for j in range(N-1):
            small_acc_traj[i*(N-1) + j] = acceleration_trajectory[i]



    small_pos_traj[0] = position_trajectory[0]
    small_vel_traj[0] = velocity_trajectory[0]

    for i in range(len(small_pos_traj)-1):
        small_delta_t = small_time_traj[i+1] - small_time_traj[i]
        small_pos_traj[i+1] = small_pos_traj[i] + small_vel_traj[i] * small_delta_t +  small_acc_traj[i] * small_delta_t ** 2 / 2
        small_vel_traj[i+1] = small_vel_traj[i] + small_acc_traj[i] * small_delta_t

    small_acc_traj = np.vstack((small_acc_traj, np.zeros(7)))
    assert np.allclose(position_trajectory[-1], small_pos_traj[-1], atol=1e-2), ERROR(position_trajectory[-1], small_pos_traj[-1])
    return small_time_traj, small_pos_traj, small_vel_traj, small_acc_traj





def visualize_trajectory_pointclouds(time_traj, 
                                     pos_traj, 
                                     vel_traj,
                                     vel_cons, 
                                     acc_cons,
                                     mid_arm_offset = 0.12, 
                                     acc_rgb = (1,0,0), 
                                     vel_rgb = (0,1,0), 
                                     normal_rgb=(0,0,1),
                                     acc_violated_rgb=(1,0,0),
                                     con_bound = 0.95
                                     ):
    arm_file_path="./models/iiwa14_david_cheap_bigger3.dmd.yaml"
    num_joints = 7
    time_step = 0.0

    assert len(time_traj)==len(pos_traj)==len(vel_traj)==len(vel_cons)==len(acc_cons)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.package_map().AddPackageXml(filename=os.path.abspath("./models/package.xml"))

    # Add arm
    parser.AddModels(arm_file_path)
    try:
        arm = plant.GetModelInstanceByName("iiwa")
    except:
        arm = plant.GetModelInstanceByName("arm")

    parser.SetAutoRenaming(True)

    num_arms = 2
    arm_positions = []
    for i in range(num_arms):
        if i == 0:
            arm_positions.append(pos_traj[i*int(len(pos_traj)/num_arms)])
        elif i == 1:
            arm_positions.append(pos_traj[ int((i+mid_arm_offset)*(len(pos_traj)/num_arms)) ])

        # add other arms
        new_iiwa = parser.AddModels("models/iiwa14_spheres_collision.urdf")
        # model isntance index
        new_wsg = parser.AddModels("models/schunk_wsg_50_welded_fingers_sphere_collision.sdf")

        # Weld iiwa to the world frame.
        plant.WeldFrames(plant.world_frame(),
                            plant.GetFrameByName("base", new_iiwa[0]),
                                RigidTransform())

        # Weld wsg to the iiwa end-effector.
        plant.WeldFrames(
                            plant.GetFrameByName("iiwa_link_7", new_iiwa[0]),
                            plant.GetFrameByName("body", new_wsg[0]),
                            RigidTransform(rpy=RollPitchYaw([np.pi / 2., 0, 0]),
                                            p=[0, 0, 0.114]))
        # break
        
    plant.Finalize()

    placeholder_trajectory = PiecewisePolynomial(np.zeros((num_joints, 1)))
    trajectory_source = builder.AddSystem(
        TrajectorySource(placeholder_trajectory, output_derivative_order=1)
    )

    # Meshcat
    meshcat = StartMeshcat()
    meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat
    )

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    # plant_context = plant.GetMyContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)

    arm_components = ArmComponents(
        num_joints=7,
        diagram=diagram,
        plant=plant,
        trajectory_source=trajectory_source,
        meshcat=meshcat,
        meshcat_visualizer=meshcat_visualizer,
        diagram_context=diagram_context
    )




    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    plant_context = arm_components.plant.GetMyContextFromRoot(context)
    arm_components.meshcat_visualizer.StartRecording()

    arm_components.meshcat.SetCameraPose([-1,0.4,1],[0,0,0.3])

    acc_con_violated_end_effector_locations = []
    acc_con_end_effector_locations = []
    vel_cons_end_effector_locations = []
    normal_end_effector_locations = []

    for i, pos in enumerate(pos_traj):
        if acc_cons[i] > 1.001:
            acc_con_violated_end_effector_locations += [pos]
        elif acc_cons[i] > con_bound:
            acc_con_end_effector_locations += [pos]
        elif vel_cons[i] > con_bound:
            vel_cons_end_effector_locations += [pos]
        else:
            normal_end_effector_locations += [pos]

    acc_con_violated_end_effector_locations = ForwardKinematics(acc_con_violated_end_effector_locations)
    acc_con_end_effector_locations = ForwardKinematics(acc_con_end_effector_locations)
    vel_cons_end_effector_locations = ForwardKinematics(vel_cons_end_effector_locations)
    normal_end_effector_locations = ForwardKinematics(normal_end_effector_locations)

    pointcloud = PointCloud(len(normal_end_effector_locations))
    pointcloud.mutable_xyzs()[:] = np.array(list(map(lambda X: X.translation(), normal_end_effector_locations))).T[:]
    r, g, b, a = normal_rgb[0], normal_rgb[1], normal_rgb[2], 1
    arm_components.meshcat.SetObject("paths/normal", pointcloud, 0.035, rgba=Rgba( r,g,b,a ) )

    pointcloud = PointCloud(len(vel_cons_end_effector_locations))
    pointcloud.mutable_xyzs()[:] = np.array(list(map(lambda X: X.translation(), vel_cons_end_effector_locations))).T[:]
    r, g, b, a = vel_rgb[0], vel_rgb[1], vel_rgb[2], 1
    arm_components.meshcat.SetObject("paths/vel", pointcloud, 0.035, rgba=Rgba( r,g,b,a ) )


    pointcloud = PointCloud(len(acc_con_end_effector_locations))
    pointcloud.mutable_xyzs()[:] = np.array(list(map(lambda X: X.translation(), acc_con_end_effector_locations))).T[:]
    r, g, b, a = acc_rgb[0], acc_rgb[1], acc_rgb[2], 1
    arm_components.meshcat.SetObject("paths/acc", pointcloud, 0.035, rgba=Rgba( r,g,b,a ) )

    pointcloud = PointCloud(len(acc_con_violated_end_effector_locations))
    pointcloud.mutable_xyzs()[:] = np.array(list(map(lambda X: X.translation(), acc_con_violated_end_effector_locations))).T[:]
    r, g, b, a = acc_violated_rgb[0], acc_violated_rgb[1], acc_violated_rgb[2], 1
    arm_components.meshcat.SetObject("paths/acc_violated", pointcloud, 0.035, rgba=Rgba( r,g,b,a ) )


    # arm_components.plant.SetPositions(plant_context, small_pos_traj[-1])
    arm_components.plant.SetPositions(plant_context, np.array([pos_traj[-1]] + arm_positions).flatten())
    # arm_components.plant.SetPositions(plant_context, np.array([small_pos_traj[-1]] + [arm_positions[0]]).flatten())
    simulator.AdvanceTo(0.01)

    # replay trajectories
    arm_components.meshcat_visualizer.StopRecording()
    arm_components.meshcat_visualizer.PublishRecording()

    return arm_components


def pickle_trajectory(name, time_traj, pos_traj, vel_traj, acc_traj, vel_cons, acc_cons, jerk_traj=None, jerk_cons=None):
    l = len(time_traj)
    assert l == len(pos_traj) == len(vel_traj) == len(acc_traj) == len(vel_cons) == len(acc_cons)
    if jerk_traj is not None:
        assert len(jerk_traj) == l
    if jerk_cons is not None:
        assert len(jerk_traj) == l
    temp_dict = dict()
    temp_dict["time_traj"] = time_traj
    temp_dict["pos_traj"] = pos_traj
    temp_dict["vel_traj"] = vel_traj
    temp_dict["acc_traj"] = acc_traj
    temp_dict["vel_cons"] = vel_cons
    temp_dict["acc_cons"] = acc_cons
    temp_dict["jerk_traj"] = jerk_traj
    temp_dict["jerk_cons"] = jerk_cons
    with open("./saved_data/" + name + ".pkl", 'wb') as f:
        pickle.dump(temp_dict, f)


def unpickle_trajectory(name):
    with open("./saved_data/" + name + ".pkl", 'rb') as f:
        example_data = pickle.load(f)
        return example_data
