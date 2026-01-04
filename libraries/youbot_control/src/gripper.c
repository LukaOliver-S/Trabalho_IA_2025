/*
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description:   Implement the functions defined in gripper.h
 */

#include "gripper.h"

#include <webots/motor.h>
#include <webots/robot.h>

#include "tiny_math.h"

#define LEFT 0
#define RIGHT 1

#define MIN_POS 0.0
#define MAX_POS 0.025
#define OFFSET_WHEN_LOCKED 0.021

static WbDeviceTag finger_left;
static WbDeviceTag finger_right;

void gripper_init() {
  finger_left = wb_robot_get_device("finger::left");
  finger_right = wb_robot_get_device("finger::right");

  if (finger_left)
    wb_motor_set_velocity(finger_left, 0.03);
  if (finger_right)
    wb_motor_set_velocity(finger_right, 0.03);
}

void gripper_grip() {
  if (finger_left)
    wb_motor_set_position(finger_left, MIN_POS);
  if (finger_right)
    wb_motor_set_position(finger_right, MIN_POS);
}

void gripper_release() {
  if (finger_left)
    wb_motor_set_position(finger_left, MAX_POS);
  if (finger_right)
    wb_motor_set_position(finger_right, MAX_POS);
}

void gripper_set_gap(double gap) {
  double v = bound(0.5 * (gap - OFFSET_WHEN_LOCKED), MIN_POS, MAX_POS);
  // Ensure the value is never negative due to floating point precision issues
  if (v < 0.0) v = 0.0;
  
  if (finger_left)
    wb_motor_set_position(finger_left, v);
  if (finger_right)
    wb_motor_set_position(finger_right, v);
}
