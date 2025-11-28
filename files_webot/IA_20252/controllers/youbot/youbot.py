from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper
import random as rand
class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        self.front_camera = self.robot.getDevice("front camera")
        self.front_camera.enable(self.time_step)

        self.left_camera = self.robot.getDevice("left camera")
        self.left_camera.enable(self.time_step)
        
        self.right_camera = self.robot.getDevice("right camera")
        self.right_camera.enable(self.time_step)

        self.back_camera = self.robot.getDevice("back camera")
        self.back_camera.enable(self.time_step)
        
        
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.time_step)
        
        
    def run(self):
 
        while self.robot.step(self.time_step) != -1:
            ranges = self.lidar.getRangeImage()


            self.base.forwards()

            
                        
            image_data = self.front_camera.getImage()
            
            
            
            if image_data:
                filename = "camera_image.png"
                self.front_camera.saveImage(filename, 100)  # 100 = quality
                print(f"Image saved as {filename}")
        raise NotImplementedError("This method should be implemented")

if __name__ == "__main__":
    controller = YouBotController()
    controller.run()