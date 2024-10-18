import typing as T

import numpy as np
import numpy.typing as npt

try:
    from tkinter import Tk, Canvas, Toplevel
except ImportError:
    from Tkinter import Tk, Canvas, Toplevel


import time
from shortest_walk_through_gcs.util import WARN, INFO, ERROR, YAY
import math

BLOCK_COLORS = ["#E3B5A4", "#E8D6CB", "#C3DFE0", "#F6E4F6", "#F4F4F4"]
# BLOCK_COLORS = ["#843B62", "#E3B5A4", "#843B62", "#F6E4F6", "#F4F4F4"]
# ARM_COLOR = "#843B62"
ARM_COLOR = "#ADDFFF"
ARM_NOT_EMPTY_COLOR = "#621940"  # 5E3886 621940

TABLE_BACKGROUND = '#F5F8E8'
TABLE_COLOR = '#8789C0'

TEXT_COLOR = "#0B032D"
BLACK = "#0B032D"
BACKGROUND = "#F5E9E2"
CELL_WIDTH = 50
CELL_WIDTH = 90

class SimpleObject:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def flip(self):
        self.w, self.h = self.h, self.w

def rotate(points, angle, center):
    angle = math.radians(angle)
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    cx, cy = center
    new_points = []
    for x_old, y_old in points:
        x_old -= cx
        y_old -= cy
        x_new = x_old * cos_val - y_old * sin_val
        y_new = x_old * sin_val + y_old * cos_val
        new_points.append([x_new + cx, y_new + cy])
    return new_points

def get_vertices(cube:SimpleObject, x,y):
    w, h = cube.w, cube.h
    return [[x-w, y-h], [x-w, y+h], [x+w,y+h], [x+w,y-h]]


class Draw2DSolution:
    def __init__(
        self,
        num_blocks: int,
        modes: T.List[str],
        segments: T.List[T.Tuple[npt.NDArray, npt.NDArray]],
        obj_half_widths: T.List[float],
        obj_half_heights: T.List[float] = None,
        h_min = 0,
        h_max = 10, 
        goal_min = 6,
        goal_max = 10,
        title = "",
        title_font = 20,
        goal_font = 20,
        fast: bool = True,
        sleep_between_actions = 0.0
    ):
        self.num_blocks = num_blocks
        if obj_half_heights is None:
            obj_half_heights = [0.5] * self.num_blocks
        self.objects = [ SimpleObject(obj_half_widths[i], obj_half_heights[i]) for i in range(self.num_blocks) ]

        self.pixel_scaling = CELL_WIDTH

        self.height_in_m = 4 # rows
        self.width_in_m = h_max-h_min # cols

        self.border_buffer = 50 # buffer to the borders
        self.width = self.width_in_m * (CELL_WIDTH) + 2 * self.border_buffer
        self.height = self.height_in_m * (CELL_WIDTH) + 2 * self.border_buffer

        self.goal_min = goal_min
        self.goal_max = goal_max

        self.modes = modes
        self.segments = segments
        self.obj_half_widths = obj_half_widths
        self.goal_font = goal_font

        if fast:
            self.speed = 6  # units/s
            self.grasp_dt = 0.2  # s
        else:
            self.speed = 2  # units/s
            self.grasp_dt = 0.3  # s

        self.move_dt = 0.025  # s

        self.grasping = False

        self.title = title
        self.title_font = title_font
        self.arm_height = 2

        # tkinter initialization
        self.tk = Tk()
        self.tk.withdraw()
        self.top = Toplevel(self.tk)
        self.top.wm_title("Moving Blocks")
        self.top.protocol("WM_DELETE_WINDOW", self.top.destroy)

        self.canvas = Canvas(self.top, width=self.width, height=self.height, background=BACKGROUND)
        self.canvas.pack()
        self.cells = {}
        self.environment = []
        self.sleep_between_actions = sleep_between_actions
        self.draw_solution()

    def draw_solution(self):
        self.draw_initial(self.segments[0][0])
        time.sleep(0.5)
        for i, mode in enumerate(self.modes):
            state_now, state_next = self.segments[i]
            self.move_from_to(state_now, state_next, mode)
            time.sleep(self.sleep_between_actions)
        time.sleep(0.5)
        self.tk.destroy()

    def draw_initial(self, state_now):
        self.grasping = False
        # must augment states
        # if "hold" in mode:
        # augment state with object heights
        full_state_left = np.zeros((self.num_blocks+1, 2))
        for i in range(self.num_blocks):
            full_state_left[i, 0] = state_now[i]
            full_state_left[i, 1] = self.objects[i].h

        full_state_left[-1, 0] = state_now[-1]
        full_state_left[-1, 1] = self.arm_height
        self.move_along_line(full_state_left, full_state_left)

    def move_from_to(self, state_now, state_next, mode):

        # must augment states
        # augment state with object heights
        full_state_left = np.zeros((self.num_blocks+1, 2))
        full_state_right = np.zeros((self.num_blocks+1, 2))
        for i in range(self.num_blocks):
            full_state_left[i, 0] = state_now[i]
            full_state_left[i, 1] = self.objects[i].h
            full_state_right[i, 0] = state_next[i]
            full_state_right[i, 1] = self.objects[i].h
        full_state_left[-1, 0] = state_now[-1]
        full_state_left[-1,1] = self.arm_height
        full_state_right[-1, 0] = state_next[-1]
        # need to fill full_state right arm height

        if mode == "empty":
            self.grasping = False
            max_height_in_between = self.arm_height
            # determine if we need to change height
            for i in range(self.num_blocks):
                if (state_now[-1] <= state_now[i] <= state_next[-1]) or (state_next[-1] <= state_now[i] <= state_now[-1]):
                    # object is in between
                    max_height_in_between = max(max_height_in_between, self.objects[i].h * 2)

            full_state_left_intermediate = full_state_left.copy()
            
            full_state_left_intermediate[-1,1] = max_height_in_between
            full_state_right[-1,1] = max_height_in_between
            self.arm_height = max_height_in_between

            self.move_along_line(full_state_left, full_state_left_intermediate)
            self.move_along_line(full_state_left_intermediate, full_state_right)

        elif "hold" in mode:
            obj_index = int(mode[-1])
            self.grasping = False
            # get down to the object
            full_state_left_down_at_object = full_state_left.copy()
            self.arm_height = self.objects[obj_index].h*2
            full_state_left_down_at_object[-1,1] = self.arm_height
            self.move_along_line(full_state_left, full_state_left_down_at_object)
            self.grasping = True
            time.sleep(self.grasp_dt)

            # move the object: up, laterally, down
            max_height_in_between = 0.0
            # determine if we need to change height
            for i in range(self.num_blocks):
                if i != obj_index:
                    if (state_now[-1] <= state_now[i] <= state_next[-1]) or (state_next[-1] <= state_now[i] <= state_now[-1]):
                        # object is in between
                        max_height_in_between = max(max_height_in_between, self.objects[i].h * 2)
            
            full_state_left_up_at_object = full_state_left_down_at_object.copy()
            full_state_left_up_at_object[obj_index, 1] += max_height_in_between
            full_state_left_up_at_object[-1, 1] += max_height_in_between
            self.move_along_line(full_state_left_down_at_object, full_state_left_up_at_object)

            # go laterally
            full_state_right_up_at_object = full_state_right.copy()
            full_state_right_up_at_object[obj_index, 1] = full_state_left_up_at_object[obj_index,1]
            full_state_right_up_at_object[-1, 1] = full_state_left_up_at_object[-1,1]
            self.move_along_line(full_state_left_up_at_object, full_state_right_up_at_object)

            # go down
            full_state_right[-1,1] = self.arm_height
            self.move_along_line(full_state_right_up_at_object, full_state_right)
            time.sleep(self.grasp_dt)
            self.grasping = False

        elif "flip" in mode:
            obj_index = int(mode[-1])
            self.grasping = False
            # get down to the object and grasph it
            full_state_left_down_at_object = full_state_left.copy()
            self.arm_height = self.objects[obj_index].h*2
            full_state_left_down_at_object[-1,1] = self.arm_height
            self.move_along_line(full_state_left, full_state_left_down_at_object)
            self.grasping = True
            time.sleep(self.grasp_dt)

            if "flip left" in mode:
                for theta in range(0,91,2):
                    vertices = get_vertices(self.objects[obj_index], full_state_left[obj_index][0], full_state_left[obj_index][1])
                    rotation_center = vertices[0]
                    rotated_vertices = rotate(vertices, theta, rotation_center)
                    arm_location = rotated_vertices[2]

                    time.sleep(self.move_dt)
                    self.clear()
                    self.draw_block_vertices(rotated_vertices, obj_index)
                    for index in range(self.num_blocks):
                        if index != obj_index:
                            self.draw_block(full_state_left[index], index)
                    self.draw_arm(arm_location)
                    self.draw_background()
                    self.tk.update()

            elif "flip right" in mode:
                for theta in range(0,-91,-2):
                    vertices = get_vertices(self.objects[obj_index], full_state_left[obj_index][0], full_state_left[obj_index][1])
                    rotation_center = vertices[3]
                    rotated_vertices = rotate(vertices, theta, rotation_center)
                    arm_location = rotated_vertices[1]
                    time.sleep(self.move_dt)
                    self.clear()
                    self.draw_block_vertices(rotated_vertices, obj_index)
                    for index in range(self.num_blocks):
                        if index != obj_index:
                            self.draw_block(full_state_left[index], index)
                    self.draw_arm(arm_location)
                    self.draw_background()
                    self.tk.update()

            else:
                ERROR(mode)
                raise NotImplementedError()
            
            self.objects[obj_index].flip()
            self.arm_height = self.objects[obj_index].h*2
            self.grasping = False
            time.sleep(self.grasp_dt)

        else:
            ERROR(mode)
            raise NotImplementedError()
        
    def move_along_line(self, state_now, state_next):  
        delta = state_next - state_now
        distance = np.linalg.norm(delta[-1])
        distance_per_dt = self.speed * self.move_dt
        num_steps = int(max(float(distance / distance_per_dt), 1.0))
        for i in range(0, num_steps + 1):
            self.draw_state(state_now + delta * i / num_steps)
            time.sleep(self.move_dt)
        

    def draw_state(self, state):
        self.clear()
        for obj_index in range(self.num_blocks):
            self.draw_block(state[obj_index], obj_index)
        self.draw_arm(state[-1])
        self.draw_background()
        self.tk.update()

    def clear(self):
        self.canvas.delete("all")

    def draw_block(self, block_state, block_num):
        x, y = self.transform_xy(block_state)
        x_side = self.pixel_scaling * self.objects[block_num].w
        y_side = self.pixel_scaling * self.objects[block_num].h

        self.cells[(x,y)] = [
            self.canvas.create_rectangle(x - x_side, y - y_side,
                                         x + x_side, y + y_side,
                                         fill=BLOCK_COLORS[block_num], 
                                         outline='black', 
                                         width=2),
            self.canvas.create_text(x, y, text=block_num, fill=TEXT_COLOR),
            ]
        
    def draw_block_vertices(self, block_vertices, block_num):
        pixel_vertices = [ self.transform_xy(xy) for xy in block_vertices ]
        center = [ sum([x[0] for x in pixel_vertices])/len(pixel_vertices), sum([x[1] for x in pixel_vertices])/len(pixel_vertices) ]

        self.cells[(pixel_vertices[0][0],pixel_vertices[0][1])] = [
            self.canvas.create_polygon(pixel_vertices, fill=BLOCK_COLORS[block_num], 
                                         outline='black', 
                                         width=2),
            self.canvas.create_text(center[0], center[1], text=block_num, fill=TEXT_COLOR),
            ]

    def draw_arm(self, arm_state):
        if self.grasping:
            arm_color = ARM_NOT_EMPTY_COLOR
        else:
            arm_color = ARM_COLOR
        
        x, y = self.transform_xy(arm_state)
        
        grasp_buffer = 3 # 0 | 3 | 5
        gripper_height = 20
        gripper_width = self.pixel_scaling

        # stem_length = 300
        stem_width = 20
        stemStart = 0

        y = y - gripper_height / 2 - grasp_buffer

        robot = [
            self.canvas.create_rectangle(x - stem_width / 2., stemStart,
                                        x + stem_width / 2., y,
                                        fill=arm_color, outline='black', width=2),
            self.canvas.create_rectangle(x - gripper_width / 2., y - gripper_height / 2.,
                                        x + gripper_width / 2., y + gripper_height / 2.,
                                        fill=arm_color, outline='black', width=2),
        ]

    def draw_background(self, bin_color=TABLE_COLOR):
        epsilon_for_view = 20
        self.environment = [
            # left wall
            self.canvas.create_rectangle(0, 0,
                                        self.border_buffer, self.height,
                                        fill=bin_color, outline='black', width=2),
            # right wall
            self.canvas.create_rectangle(self.width - self.border_buffer, 0,
                                        self.width + epsilon_for_view, self.height,
                                        fill=bin_color, outline='black', width=2),
            # bottom wall
            self.canvas.create_rectangle(0, self.height - self.border_buffer,
                                        self.width+epsilon_for_view, self.height + epsilon_for_view,
                                        fill=bin_color, outline='black', width=2),

            # draw goal region
            self.canvas.create_rectangle(self.transform_x_to_pixels(self.goal_min), self.height - self.border_buffer,
                                        self.transform_x_to_pixels(self.goal_max), self.height + epsilon_for_view,
                                        fill="#C7FFD8", outline='black', width=2),
            self.canvas.create_text(self.transform_x_to_pixels((self.goal_min+self.goal_max)/2), self.height-self.border_buffer/2, text="target region", fill=TEXT_COLOR, font = ("Helvetica", self.goal_font)),


            self.canvas.create_rectangle(0, 0,
                                        self.width+epsilon_for_view, self.border_buffer + 5,
                                        fill=bin_color, outline='black', width=2),

            self.canvas.create_text(self.width/2, (self.border_buffer + 5)/2, text=self.title, fill=TEXT_COLOR,
                                    font = ("Helvetica", self.title_font)),

            # self.canvas.create_rectangle(self.table_x1 + self.table_width - self.border_buffer, self.table_y1 + self.pixel_scaling*3,
            #                              self.table_x1 + self.table_width+20, self.table_y1 + self.table_height + 500,
            #                              fill=bin_color, outline='black', width=2),
            
        ]


    def transform_y_to_pixels(self, y):
        """
        tranform row number into y location
        """
        return self.border_buffer + (self.height_in_m-y) * CELL_WIDTH # why is this 10 here?

    def transform_x_to_pixels(self, x):
        """
        tranform column number into x location
        """
        return (x) * (CELL_WIDTH) + self.border_buffer

    def transform_xy(self, xy):
        """
        tranform a [col, row] array into [x,y]
        """
        return [ self.transform_x_to_pixels(xy[0]), self.transform_y_to_pixels(xy[1]) ]