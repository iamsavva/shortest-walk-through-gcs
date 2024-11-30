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
from PIL import ImageGrab
from PIL import Image, ImageTk

from shortest_walk_through_gcs.util_plotting import overlay_colors

BLOCK_COLORS = ["#E3B5A4", "#E8D6CB", "#C3DFE0", "#F6E4F6", "#F4F4F4"]
# BLOCK_COLORS = ["#843B62", "#E3B5A4", "#843B62", "#F6E4F6", "#F4F4F4"]
# ARM_COLOR = "#843B62"
ARM_COLOR = "#ADDFFF"
ARM_NOT_EMPTY_COLOR = "#621940"  # 5E3886 621940

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
        sleep_between_actions = 0.0,
        height_in_m = 4,
        border_buffer = 50,
        draw_top=True
    ):
        self.num_blocks = num_blocks
        if obj_half_heights is None:
            obj_half_heights = [0.5] * self.num_blocks
        self.objects = [ SimpleObject(obj_half_widths[i], obj_half_heights[i]) for i in range(self.num_blocks) ]

        self.pixel_scaling = CELL_WIDTH

        self.height_in_m = height_in_m # rows
        self.width_in_m = h_max-h_min # cols

        self.border_buffer = border_buffer # buffer to the borders
        self.width = self.width_in_m * (CELL_WIDTH) + 2 * self.border_buffer
        self.height = self.height_in_m * (CELL_WIDTH) + 2 * self.border_buffer
        if not draw_top:
            self.height -= self.border_buffer
        self.draw_top=draw_top

        self.goal_min = goal_min
        self.goal_max = goal_max

        self.modes = modes[1:-1]
        self.segments = segments[1:-1]

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
        self.tk = Tk() # type: Tk
        self.tk.withdraw()
        self.top = Toplevel(self.tk)
        self.top.wm_title(self.title)
        self.top.protocol("WM_DELETE_WINDOW", self.top.destroy)

        self.canvas = Canvas(self.top, width=self.width, height=self.height, background=BACKGROUND)
        self.canvas.pack()
        self.images = []
        self.sleep_between_actions = sleep_between_actions
        self.shadow_counter = 0

    def save_into_png(self, screenshot_name="temp", offset=28):
        x = self.tk.winfo_rootx()-offset
        y = self.tk.winfo_rooty()-offset
        width = self.width
        height = self.height
        takescreenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
        takescreenshot.save(screenshot_name+".png")

    def draw_solution(self, shadow=None):
        self.draw_initial(self.segments[0][0])
        time.sleep(0.5)
        for i, mode in enumerate(self.modes):
            state_now, state_next = self.segments[i]
            self.move_from_to(state_now, state_next, mode, i, shadow)
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

    def move_from_to(self, state_now, state_next, mode, ind, shadow=None):

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

            all_states = []
            if not np.allclose(full_state_left, full_state_left_intermediate):
                all_states += self.move_along_line(full_state_left, full_state_left_intermediate, shadow)
            all_states += self.move_along_line(full_state_left_intermediate, full_state_right, shadow)

            # get down to the next object
            if ind+1 < len(self.modes):
                # get down to the object
                next_mode = self.modes[ind+1]
                obj_index = int(next_mode[-1])
                full_state_right_down_at_object = full_state_right.copy()
                self.arm_height = self.objects[obj_index].h*2
                full_state_right_down_at_object[-1,1] = self.arm_height
                all_states += self.move_along_line(full_state_right, full_state_right_down_at_object, shadow)
            if shadow is not None:
                self.draw_shadow_states(all_states, shadow)
            time.sleep(self.grasp_dt)

            

        elif "hold" in mode:
            obj_index = int(mode[-1])
            
            # get down to the object
            full_state_left_down_at_object = full_state_left.copy()
            self.arm_height = self.objects[obj_index].h*2
            full_state_left_down_at_object[-1,1] = self.arm_height
            self.grasping = True

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
            all_states=[]
            if not np.allclose(full_state_left_down_at_object, full_state_left_up_at_object):
                all_states +=  self.move_along_line(full_state_left_down_at_object, full_state_left_up_at_object, shadow)

            # go laterally
            full_state_right_up_at_object = full_state_right.copy()
            full_state_right_up_at_object[obj_index, 1] = full_state_left_up_at_object[obj_index,1]
            full_state_right_up_at_object[-1, 1] = full_state_left_up_at_object[-1,1]
            all_states +=  self.move_along_line(full_state_left_up_at_object, full_state_right_up_at_object, shadow)

            # go down
            full_state_right[-1,1] = self.arm_height
            if not np.allclose(full_state_right_up_at_object, full_state_right):
                all_states +=  self.move_along_line(full_state_right_up_at_object, full_state_right, shadow)
            if shadow is not None:
                self.draw_shadow_states(all_states, shadow)
            time.sleep(self.grasp_dt)
            self.grasping = False

        elif "flip" in mode:
            obj_index = int(mode[-1])
            
            # get down to the object and grasph it
            full_state_left_down_at_object = full_state_left.copy()
            self.arm_height = self.objects[obj_index].h*2
            full_state_left_down_at_object[-1,1] = self.arm_height
            self.grasping = True
            # time.sleep(self.grasp_dt)

            if "flip left" in mode:
                if shadow is None:
                    for theta in range(0,91,2):
                        vertices = get_vertices(self.objects[obj_index], full_state_left[obj_index][0], full_state_left[obj_index][1])
                        rotation_center = vertices[0]
                        rotated_vertices = rotate(vertices, theta, rotation_center)
                        arm_location = rotated_vertices[2]

                        time.sleep(self.move_dt)
                        self.clear()
                        self.draw_background()
                        self.draw_block_vertices(rotated_vertices, obj_index)
                        for index in range(self.num_blocks):
                            if index != obj_index:
                                self.draw_block(full_state_left[index], index)
                        self.draw_arm(arm_location)
                        self.tk.update()
                else:
                    self.clear()
                    for index in range(self.num_blocks):
                        if index != obj_index:
                            self.draw_block(full_state_left[index], index)

                    for i, alpha in enumerate(shadow):
                        theta = i * int(90 // len(shadow))
                        if i == len(shadow)-1:
                            theta = 90
                        vertices = get_vertices(self.objects[obj_index], full_state_left[obj_index][0], full_state_left[obj_index][1])
                        rotation_center = vertices[0]
                        rotated_vertices = rotate(vertices, theta, rotation_center)
                        arm_location = rotated_vertices[2]    
                        self.draw_block_vertices(rotated_vertices, obj_index, alpha)
                        self.draw_arm(arm_location, alpha)

                    self.draw_background()
                    self.tk.update()
                    self.save_into_png(self.title + "_"+ str(self.shadow_counter))
                    self.shadow_counter+=1


            elif "flip right" in mode:
                if shadow is None:
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
                    self.clear()
                    for index in range(self.num_blocks):
                        if index != obj_index:
                            self.draw_block(full_state_left[index], index)

                    for i, alpha in enumerate(shadow):
                        theta = -i * int(90 // len(shadow))
                        if i == len(shadow)-1:
                            theta = -90
                        vertices = get_vertices(self.objects[obj_index], full_state_left[obj_index][0], full_state_left[obj_index][1])
                        rotation_center = vertices[3]
                        rotated_vertices = rotate(vertices, theta, rotation_center)
                        arm_location = rotated_vertices[1]    
                        self.draw_block_vertices(rotated_vertices, obj_index, alpha)
                        self.draw_arm(arm_location, alpha)

                    self.draw_background()
                    self.tk.update()
                    self.save_into_png(self.title + "_"+ str(self.shadow_counter))
                    self.shadow_counter+=1


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
        
    def move_along_line(self, state_now, state_next, shadow=None):  
        list_of_states = []
        delta = state_next - state_now
        distance = np.linalg.norm(delta[-1])
        distance_per_dt = self.speed * self.move_dt
        num_steps = int(max(float(distance / distance_per_dt), 1.0))
        for i in range(0, num_steps + 1):
            list_of_states.append(state_now + delta * i / num_steps)

        # shadow is none -- draw them
        if shadow is None:
            for state in list_of_states:
                self.clear()
                self.draw_state(state)
                self.draw_background()
                self.tk.update()
                time.sleep(self.move_dt)
        return list_of_states
    
    def draw_shadow_states(self, states, shadow):
        def select_uniformly_spaced(K, N):
            if N >= len(K):
                return K  # If N is greater than or equal to the size of K, return the entire list
            indices = [(i * int((len(K) - 1) / (N - 1))) for i in range(N)]
            indices[-1] = len(K)-1
            return [K[idx] for idx in indices]
        
        states_to_plot = select_uniformly_spaced(states, len(shadow))
        self.clear()

        for i in range(0, len(states_to_plot)):
            self.draw_state(states_to_plot[i], shadow[ len(shadow)-len(states_to_plot)+i ])

        self.draw_background()
        self.tk.update()
        self.save_into_png(self.title + "_"+ str(self.shadow_counter))
        self.shadow_counter+=1
        time.sleep(0.75)

    def draw_state(self, state, alpha=1):
        for obj_index in range(self.num_blocks):
            self.draw_block(state[obj_index], obj_index, alpha)
        self.draw_arm(state[-1], alpha)
        
    def clear(self):
        self.canvas.delete("all")
        self.images = []

    def draw_block(self, block_state, block_num, alpha=1):
        x, y = self.transform_xy(block_state)
        x_side = self.pixel_scaling * self.objects[block_num].w
        y_side = self.pixel_scaling * self.objects[block_num].h

        self.canvas.create_rectangle(x - x_side, y - y_side,
                                     x + x_side, y + y_side,
                                        fill=overlay_colors(BACKGROUND, BLOCK_COLORS[block_num], alpha), 
                                        outline=overlay_colors(BACKGROUND, "#000000", alpha),
                                        width=2)
        self.canvas.create_text(x, y, text=block_num, fill=overlay_colors(BACKGROUND, TEXT_COLOR, alpha))
        
    
    def draw_block_vertices(self, block_vertices, block_num, alpha=1):
        pixel_vertices = [ self.transform_xy(xy) for xy in block_vertices ]
        center = [ sum([x[0] for x in pixel_vertices])/len(pixel_vertices), sum([x[1] for x in pixel_vertices])/len(pixel_vertices) ]

        self.canvas.create_polygon(pixel_vertices, fill=overlay_colors(BACKGROUND, BLOCK_COLORS[block_num], alpha), 
                                        outline=overlay_colors(BACKGROUND, "#000000", alpha), 
                                        width=2)
        self.canvas.create_text(center[0], center[1], text=block_num, fill=TEXT_COLOR)

    def draw_arm(self, arm_state, alpha=1):
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

        self.canvas.create_rectangle(x - stem_width / 2., stemStart,
                                    x + stem_width / 2., y,
                                    fill=overlay_colors(BACKGROUND, arm_color, alpha), outline=overlay_colors(BACKGROUND, "#000000", alpha), width=2)
        
        self.canvas.create_rectangle(x - gripper_width / 2., y - gripper_height / 2.,
                                    x + gripper_width / 2., y + gripper_height / 2.,
                                    fill=overlay_colors(BACKGROUND, arm_color, alpha), outline=overlay_colors(BACKGROUND, "#000000", alpha), width=2)
        

    def draw_background(self, bin_color=TABLE_COLOR):
        epsilon_for_view = 20
        # left wall
        self.canvas.create_rectangle(-epsilon_for_view, -epsilon_for_view,
                                    self.border_buffer, self.height+epsilon_for_view,
                                    fill=bin_color, outline='black', width=2)
        # right wall
        self.canvas.create_rectangle(self.width - self.border_buffer, -epsilon_for_view,
                                    self.width + epsilon_for_view, self.height,
                                    fill=bin_color, outline='black', width=2)
        # bottom wall
        self.canvas.create_rectangle(-epsilon_for_view, self.height - self.border_buffer,
                                    self.width+epsilon_for_view, self.height + epsilon_for_view,
                                    fill=bin_color, outline='black', width=2)
        if self.draw_top:
            # top wall
            self.canvas.create_rectangle(0, 0,
                                        self.width+epsilon_for_view, self.border_buffer,
                                        fill=bin_color, outline='black', width=2)
        # draw goal region
        self.canvas.create_rectangle(self.transform_x_to_pixels(self.goal_min), self.height - self.border_buffer,
                                    self.transform_x_to_pixels(self.goal_max), self.height + epsilon_for_view,
                                    fill="#C7FFD8", outline='black', width=2)
        # draw goal region text
        self.canvas.create_text(self.transform_x_to_pixels((self.goal_min+self.goal_max)/2), self.height-self.border_buffer/2, text="target region", fill=TEXT_COLOR, font = ("Helvetica", self.goal_font))
        
        # draw title
        # self.canvas.create_text(self.width/2, (self.border_buffer + 5)/2, text=self.title, fill=TEXT_COLOR,
        #                         font = ("Helvetica", self.title_font))


    def transform_y_to_pixels(self, y):
        """
        tranform row number into y location
        """
        border_buffer = 0
        if self.draw_top:
            border_buffer +=self.border_buffer
        return border_buffer + (self.height_in_m-y) * CELL_WIDTH # why is this 10 here?

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