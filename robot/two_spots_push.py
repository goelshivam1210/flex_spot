"""
Script using the new Spot class to push a box

Author: Tim
Date: July 1, 2025
"""
import threading

from bosdyn.client.lease import LeaseKeepAlive

from spot.spot import Spot, SpotPerception

HAND_IMG_SRC = "hand_color_image"
HAND_DEPTH_SRC = "hand_depth_in_hand_color_frame"

def push_thread(spot, grasp_pt, stop_event, sync_barrier, grasp_event, push_event):

    while not stop_event.is_set():
        # spot.start()

        with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
            # spot.power_on()
            # spot.stand_up()

            # # 1. Take picture of box and get grasp point
            # spot.open_gripper()
            # color_img, depth_img = spot.take_picture(
            #     color_src=HAND_IMG_SRC,
            #     depth_src=HAND_DEPTH_SRC,
            #     save_images=True
            # )
            # # grasp_pt = SpotPerception.get_vertical_edge_grasp_point(
            # #     color_img, depth_img, spot.id, save_img=True
            # # )
            # grasp_pt = SpotPerception.get_target_from_user(color_img)

            # if not grasp_pt:
            #     print("No grasp point found")
            #     return

            # if stop_event.is_set():
            #     return
            # sync_barrier.wait()
            # if stop_event.is_set():
            #     return

            # Save current pose
            saved_yaw = spot.save_initial_yaw()

            # 2. Grasp edge
            spot.grasp_edge(grasp_pt)
            spot.open_gripper()  # Keep gripper open to prevent losing grip

            # Return to original orientation (hopefully)
            spot.return_to_saved_yaw(saved_yaw)

            if stop_event.is_set():
                return
            sync_barrier.wait()
            if stop_event.is_set():
                return
            
            dx = 1
            dy = 0 #-0.25
            if spot.id == "Psi":
                dy = -dy

            # # 3. Push box
            spot.push_object(dx=dx, dy=dy)

if __name__ == "__main__":
    phi = Spot(id="Phi", hostname="192.168.1.100")
    psi = Spot(id="Psi", hostname="192.168.1.101")

    phi.start()
    psi.start()

    phi.power_on()
    psi.power_on()

    phi.stand_up()
    psi.stand_up()

    phi.open_gripper()
    psi.open_gripper()

    color_img, depth_img = phi.take_picture(
        color_src=HAND_IMG_SRC,
        depth_src=HAND_DEPTH_SRC,
        save_images=True
    )
    phi_grasp_pt = SpotPerception.get_target_from_user(color_img)
    # phi_grasp_pt = None

    color_img, depth_img = psi.take_picture(
        color_src=HAND_IMG_SRC,
        depth_src=HAND_DEPTH_SRC,
        save_images=True
    )
    psi_grasp_pt = SpotPerception.get_target_from_user(color_img)
    # psi_grasp_pt = None

    stop_event = threading.Event()
    grasp_event = threading.Event()
    push_event = threading.Event()

    num_parties = 3
    sync_barrier = threading.Barrier(num_parties)

    phi_thread = threading.Thread(
        target=push_thread,
        args=(phi, phi_grasp_pt, stop_event, sync_barrier, grasp_event, push_event)
    )
        
    psi_thread = threading.Thread(
        target=push_thread,
        args=(psi, psi_grasp_pt, stop_event, sync_barrier, grasp_event, push_event)
    )

    print("Starting both robots")
    phi_thread.start()
    psi_thread.start()

    try:
        print("Waiting for both spots to locate grasp points")
        # input("Press Enter to grab points...")
        # grasp_event.set()
        # sync_barrier.wait()

        print("Waiting for both spots to grab the box")
        input("Press Enter to begin pushing...")
        sync_barrier.wait()
        # push_event.set()
        print("Pushing")
        phi_thread.join()
        psi_thread.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected! Stopping all robots...")
        stop_event.set()
        # Unblock all threads waiting on an Event, so they can check stop_event and exit
        grasp_event.set()
        push_event.set()
        phi_thread.join()
        psi_thread.join()
        print("All threads stopped.")
