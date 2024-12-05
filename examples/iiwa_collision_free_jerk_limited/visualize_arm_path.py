from typing import Sequence, List, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (IllustrationProperties, MeshcatVisualizer,
                              MeshcatVisualizerParams, Rgba, RoleAssign, Role,
                              SceneGraph)
from pydrake.geometry.optimization import VPolytope, HPolyhedron
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.perception import PointCloud
from pydrake.all import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import TrajectorySource, Multiplexer, ConstantVectorSource
from pydrake.multibody.tree import RevoluteJoint
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.analysis import Simulator
from pydrake.multibody import inverse_kinematics
from pydrake.solvers import Solve

from arm_visualization import ArmComponents, arm_components_loader
from manipulation.utils import ConfigureParser

import os

def GcsDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/iiwa_collision_free_jerk_limited"

def FindModelFile(filename):
    return os.path.join(GcsDir(), filename)



def ForwardKinematics(q_list: List[Sequence[float]]) -> List[RigidTransform]:
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
            plant.EvalBodyPoseInWorld(plant_context,
                                      plant.GetBodyByName("body")))

    return X_list



def visualize_trajectory(meshcat,
                         pos_trajectory,
                         vel_trajectory,
                         time_trajectory,
                         color_trajectory,
                         robot_configurations = None,
                         ) -> None:
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    parser = Parser(plant, scene_graph)
    ConfigureParser(parser)
    parser.package_map().AddPackageXml(filename=os.path.abspath("./models/package.xml"))

    parser = Parser(plant, scene_graph)
    directives_file = FindModelFile("models/iiwa14_welded_gripper_bigger3.yaml")
    directives = LoadModelDirectives(directives_file)
    mds = ProcessModelDirectives(directives, plant, parser)

    iiwa = mds[0]
    wsg = mds[1]
    

    # Set transparency of main arm and gripper.
    # set_transparency_of_models(plant,
    #                            [iiwa.model_instance, wsg.model_instance],
    #                            transparency_arm, scene_graph)

    # Add static configurations of the iiwa for visalization.
    if robot_configurations is not None:
        # iiwa_file = FindResourceOrThrow("drake_models/iiwa_description/urdf/iiwa14_spheres_collision.urdf")
        # iiwa_file = FindModelFile("models/iiwa14_spheres_collision.urdf")
        # wsg_file = FindModelFile("models/schunk_wsg_50_welded_fingers.sdf")
        parser.SetAutoRenaming(True)

        for i, q in enumerate(robot_configurations):
            # Add iiwa and wsg for visualization.
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
            
            # body = plant.get_body(BodyIndex(i))
            # if body.name() == "world":
            #     continue
            # meshcat_body_path = os.path.join(
            #     "visualizer", *body.scoped_name().to_string().split("::")
            # )
            # for idx in plant.GetVisualGeometriesForBody(body):
            #     meshcat_path = os.path.join(
            #         meshcat_body_path, str(idx.get_value()), "<object>"
            #     )
            

            # TODO: how to set transparency?
            # meshcat.SetProperty(PATH, "opacity", transparency)
            # meshcat.SetProperty(PATH, "opacity", transparency)

            # set_transparency_of_models(plant, [new_iiwa, new_wsg],
            #                            transparency, scene_graph)

            # TODO: where to get YCB dataset?

    plant.Finalize()
    return 0

    # Set default joint angles.
    if robot_configurations:
        plant.SetDefaultPositions(
            np.hstack([np.zeros(7)] + robot_configurations))
    else:
        plant.SetDefaultPositions(np.zeros(7))

    # Add the trajectory source to the diagram.
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(),
                    scene_graph.get_source_pose_port(plant.get_source_id()))

    traj_system = builder.AddSystem(TrajectorySource(combined_traj))

    mux = builder.AddSystem(
        Multiplexer([7 for _ in range(1 + len(robot_configurations))]))
    builder.Connect(traj_system.get_output_port(), mux.get_input_port(0))

    if robot_configurations is not None:
        for i, q in enumerate(robot_configurations):
            ghost_pos = builder.AddSystem(ConstantVectorSource(q))
            builder.Connect(ghost_pos.get_output_port(),
                            mux.get_input_port(1 + i))

    builder.Connect(mux.get_output_port(), to_pose.get_input_port())

    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.delete_on_initialization_event = False
    meshcat_params.role = Role.kIllustration
    visalizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat,
                                               meshcat_params)
    meshcat.Delete()

    for i in range(7):
        meshcat_path = "drake/visualizer/iiwa14/iiwa_link_"+str(i) +"/iiwa14/Mesh/<object>"
        meshcat.SetProperty(meshcat_path, "opacity", 0.4)
    diagram = builder.Build()

    # meshcat.SetProperty("/Lights/DirectionalLight/<object>", "intensity", 1.0)
    meshcat.SetProperty("/Lights/FillLight/<object>", "intensity", 1.0)
    for i in range(7):
        meshcat_path = "drake/visualizer/iiwa14/iiwa_link_"+str(i) +"/iiwa14/Mesh/<object>"
        meshcat.SetProperty(meshcat_path, "modulated_opacity", 0.4)

    if show_path:
        X_lists = []
        for traj in trajectories:
            X_list = ForwardKinematics(
                traj.vector_values(
                    np.linspace(traj.start_time(), traj.end_time(),
                                15000)).T.tolist())
            X_lists.append(X_list)

        c_list_rgb = [[i / 255 for i in (0, 0, 255, 255)],
                      [i / 255 for i in (255, 191, 0, 255)],
                      [i / 255 for i in (255, 64, 0, 255)],
                      [i / 255 for i in (0, 200, 0, 255)],
                      ]

        for i, X_list in enumerate(X_lists):
            pointcloud = PointCloud(len(X_list))
            pointcloud.mutable_xyzs()[:] = np.array(
                list(map(lambda X: X.translation(), X_list))).T[:]
            meshcat.SetObject("paths/" + str(i),
                              pointcloud,
                              0.015 + i * 0.005,
                              rgba=Rgba(*c_list_rgb[i%4]))

    if regions is not None:
        reg_colors = plt.cm.viridis(np.linspace(0, 1, len(regions)))
        reg_colors[:, 3] = 1.0

        for i, reg in enumerate(regions):
            X_reg = ForwardKinematics(VPolytope(reg).vertices().T)
            pointcloud = PointCloud(len(X_reg))
            pointcloud.mutable_xyzs()[:] = np.array(
                list(map(lambda X: X.translation(), X_reg))).T[:]
            meshcat.SetObject("regions/" + str(i),
                              pointcloud,
                              0.015,
                              rgba=Rgba(*reg_colors[i]))
            

    simulator = Simulator(diagram)
    simulator.AdvanceTo(combined_traj.end_time())

    return meshcat