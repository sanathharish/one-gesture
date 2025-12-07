# core/actions.py

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyController, Key
import time
import numpy as np


class ActionEngine:
    def __init__(self, screen_w=1920, screen_h=1080):
        self.mouse = MouseController()
        self.keyboard = KeyController()

        self.screen_w = screen_w
        self.screen_h = screen_h

        self.dragging = False
        self.last_click_time = 0

    # -----------------------------------------------------------
    # MOUSE MOVEMENT (AIR MOUSE)
    # -----------------------------------------------------------
    def move_mouse_rel(self, cx, cy, smooth=True):
        """
        cx, cy = relative coordinates (-1 to +1 range)
        Converted to screen space
        """

        # Map to useful screen range
        x = np.interp(cx, [-0.7, 0.7], [0, self.screen_w])
        y = np.interp(cy, [-0.5, 0.5], [0, self.screen_h])

        if smooth:
            # Smooth movement by easing toward target
            current = np.array(self.mouse.position)
            target = np.array([x, y])

            new = current + (target - current) * 0.25
            self.mouse.position = (int(new[0]), int(new[1]))
        else:
            self.mouse.position = (int(x), int(y))

    # -----------------------------------------------------------
    # MOUSE CLICKS
    # -----------------------------------------------------------
    def left_click(self):
        self.mouse.click(Button.left)

    def right_click(self):
        self.mouse.click(Button.right)

    # -----------------------------------------------------------
    # CLICK + DRAG
    # -----------------------------------------------------------
    def start_drag(self):
        if not self.dragging:
            self.mouse.press(Button.left)
            self.dragging = True

    def end_drag(self):
        if self.dragging:
            self.mouse.release(Button.left)
            self.dragging = False

    # -----------------------------------------------------------
    # SCROLLING
    # -----------------------------------------------------------
    def scroll_up(self):
        self.mouse.scroll(0, 2)

    def scroll_down(self):
        self.mouse.scroll(0, -2)

    # -----------------------------------------------------------
    # VOLUME CONTROL
    # -----------------------------------------------------------
    def volume_up(self):
        self.keyboard.press(Key.media_volume_up)
        self.keyboard.release(Key.media_volume_up)

    def volume_down(self):
        self.keyboard.press(Key.media_volume_down)
        self.keyboard.release(Key.media_volume_down)

    # -----------------------------------------------------------
    # ZOOM (CTRL + SCROLL)
    # -----------------------------------------------------------
    def zoom_in(self):
        with self.keyboard.pressed(Key.ctrl):
            self.mouse.scroll(0, 2)

    def zoom_out(self):
        with self.keyboard.pressed(Key.ctrl):
            self.mouse.scroll(0, -2)

    # -----------------------------------------------------------
    # ALT TAB SYSTEM
    # -----------------------------------------------------------
    def next_window(self):
        with self.keyboard.pressed(Key.alt):
            self.keyboard.press(Key.tab)
            self.keyboard.release(Key.tab)

    def previous_window(self):
        with self.keyboard.pressed(Key.alt):
            self.keyboard.press(Key.shift)
            self.keyboard.press(Key.tab)
            self.keyboard.release(Key.tab)
            self.keyboard.release(Key.shift)
