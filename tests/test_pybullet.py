import pybullet as p
from time import sleep
import pybullet_data
import numpy as np
import time
import os

try:
    # This setup is for finding assets in a structured project
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Fallback for running the script directly
    PROJECT_ROOT = os.getcwd()

# Define paths to your URDF files
SIMPLE_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "Urdf", "box_gripper", "simple_box.urdf")
SIMPLE_URDF_PATH_2 = os.path.join(PROJECT_ROOT, "assets", "Urdf", "box_gripper", "simple_box_2.urdf")

# --- 1. Initialization ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 1])

# --- 2. Load objects ---
p.loadURDF("plane.urdf")

# Load two rigid boxes as grippers
# useMaximalCoordinates=True is more stable for kinematic control
boxId = p.loadURDF(SIMPLE_URDF_PATH, [0, 0, 0], useMaximalCoordinates=True)
boxId2 = p.loadURDF(SIMPLE_URDF_PATH_2, [0.0, 0, 0], useMaximalCoordinates=True)

# Load cloth
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_SAMPLE_ROOT = os.path.join(REPO_ROOT, "data", "sample")
CLOTH_MESH_ROOT = os.environ.get("RGBENCH_CLOTH_MESH_ROOT", os.path.join(DEFAULT_SAMPLE_ROOT, "cloth_meshes"))
model_path = os.environ.get(
    "RGBENCH_CLOTH_MESH",
    os.path.join(CLOTH_MESH_ROOT, "LargeT_Flat_Simple_5k.obj"),
)
clothId = p.loadSoftBody(
    model_path,
    basePosition=[0.1, 0, 0.01],
    scale=0.5,
    mass=1.0,  # total cloth mass
    useNeoHookean=1,
    useBendingSprings=1,
    useMassSpring=0,
    springElasticStiffness=40,  # stiffness (hardness)
    springDampingStiffness=0.1,  # damping coefficient
    useSelfCollision=0,
    frictionCoeff=0.5,
    useFaceContact=1
)
p.changeVisualShape(clothId, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)

# --- 3. Set initial pose and create anchors ---

# Move boxes to preset initial grasp poses
target_initial_pos_1 = [0.15810399, 0.10623945, 0.00415263]
target_initial_pos_2 = [0.15272144, -0.14623941, 0.00301287]
p.resetBasePositionAndOrientation(boxId, target_initial_pos_1, [0, 0, 0, 1])
p.resetBasePositionAndOrientation(boxId2, target_initial_pos_2, [0, 0, 0, 1])

# Step once so all object poses are updated
p.stepSimulation()

# Create anchors so cloth vertices "stick" to the boxes
# Args: (softBodyId, vertexIndex, rigidBodyId, linkIndex, localOffset)
# -1 means base link; [0,0,0] means attach to link center
p.createSoftBodyAnchor(clothId, 322, boxId, -1, [0, 0, 0.0])
p.createSoftBodyAnchor(clothId, 2405, boxId2, -1, [0, 0, 0.0])
print("Anchors created.")

# Capture initial box positions for subsequent animation
initial_box_pos, _ = p.getBasePositionAndOrientation(boxId)
initial_box_pos2, _ = p.getBasePositionAndOrientation(boxId2)

print("Initial box position 1:", np.round(initial_box_pos, 4))
print("Initial box position 2:", np.round(initial_box_pos2, 4))

# --- 4. Main loop ---
print("Starting active-control simulation...")
start_time = time.time()
frame_counter = 0
while p.isConnected():
    elapsed_time = time.time() - start_time

    # a. Compute a simple upward target position
    target_z = initial_box_pos[2] + 0.05 * elapsed_time  # upward velocity
    target_pos_1 = np.array([initial_box_pos[0], initial_box_pos[1], target_z])
    target_pos_2 = np.array([initial_box_pos2[0], initial_box_pos2[1], target_z])

    # b. Use kinematic control to move the boxes
    _, current_orn = p.getBasePositionAndOrientation(boxId)  # keep orientation unchanged
    p.resetBasePositionAndOrientation(boxId, target_pos_1, current_orn)
    p.resetBasePositionAndOrientation(boxId2, target_pos_2, current_orn)

    # c. Step simulation
    p.stepSimulation()

    # ================================================================
    # ===================  Collect feedback below  ===================
    # ================================================================

    # Print every few frames to avoid flooding the console
    if frame_counter % 15 == 0:
        # 1. Read actual box positions
        actual_pos_box1, _ = p.getBasePositionAndOrientation(boxId)
        actual_pos_box2, _ = p.getBasePositionAndOrientation(boxId2)

        # 2. Read actual cloth vertex positions
        mesh_data = p.getMeshData(clothId, flags=p.MESH_DATA_SIMULATION_MESH)
        if mesh_data and mesh_data[0] > 0:
            all_vertices = np.array(mesh_data[1])
            actual_pos_vertex1 = all_vertices[322]
            actual_pos_vertex2 = all_vertices[2405]

            # 3. Compute distance lag between box and vertex
            lag_distance1 = np.linalg.norm(np.array(actual_pos_box1) - actual_pos_vertex1)
            lag_distance2 = np.linalg.norm(np.array(actual_pos_box2) - actual_pos_vertex2)

            # 4. Print everything
            print(f"--- [Time: {elapsed_time:.2f}s] ---")
            print(
                f"Box 1   | Pos: {np.round(actual_pos_box1, 4)} | Vtx 322 Pos: {np.round(actual_pos_vertex1, 4)} | Lag: {lag_distance1:.6f}")
            print(
                f"Box 2   | Pos: {np.round(actual_pos_box2, 4)} | Vtx 2405 Pos: {np.round(actual_pos_vertex2, 4)} | Lag: {lag_distance2:.6f}")

    frame_counter += 1
    # ===================  end of feedback code  ===================

    # Throttle refresh rate to keep visualization smooth
    sleep(1. / 240.)
