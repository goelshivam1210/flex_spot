"""
Script using the new Spot class to push a box

Author: Tim
Date: September 8, 2025
"""
import threading
import numpy as np

from bosdyn.client.lease import LeaseKeepAlive

from spot.spot import Spot, SpotPerception

HAND_IMG_SRC = "hand_color_image"
HAND_DEPTH_SRC = "hand_depth_in_hand_color_frame"

if __name__ == "__main__":
    psi = Spot(id="Psi", hostname="192.168.1.101")
    # psi = Spot(id="Psi", hostname="192.168.80.3")
    psi.start()
    psi.power_on()
    psi.stand_up()
    psi.open_gripper()

    color_img, depth_img = psi.take_picture(
        color_src=HAND_IMG_SRC,
        depth_src=HAND_DEPTH_SRC,
        save_images=True
    )

    # psi_grasp_pt = SpotPerception.find_grasp_sam(color_img, depth_img, left=True)
    psi_grasp_pt = SpotPerception.get_red_object_center_of_mass(color_img)
    