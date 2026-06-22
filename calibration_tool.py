import cv2
import numpy as np
import time

# --- CONFIGURATION ---
# Use the index that worked for your live feed (likely 0 or 1)
CAMERA_INDEX = 1
BOARD_SIZE_PX = 640 # Destination size (matches YOLO input)
CHESS_GRID = (7, 7) # Number of internal corners (8x8 board has 7x7 intersections)
# ---

# 1. Setup Camera
# Use the simple index that worked in your detector script
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Camera not open. Check connection and index.")
    exit()

print("--- CALIBRATION TOOL ---")
print("1. Ensure the board is flat and fully visible.")
print("2. Press 'C' to capture and find the corners.")
print("3. Press 'S' to save the Homography Matrix and exit.")
print("4. Press 'Q' to quit.")

source_points_final = None 

while True:
    ret, frame = cap.read()
    if not ret: 
        time.sleep(0.01)
        continue

    # Convert to grayscale for corner detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. FIND INTERNAL CORNERS (7x7 array of intersections)
    # Using robust flags for better detection
    ret, corners = cv2.findChessboardCorners(
        gray, 
        CHESS_GRID, 
        None, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    frame_display = frame.copy()
    
    if ret:
        # Refine corner locations for sub-pixel accuracy
        cv2.cornerSubPix(
            gray, 
            corners, 
            (11, 11), 
            (-1, -1), 
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        # Draw the found inner corners (7x7)
        cv2.drawChessboardCorners(frame_display, CHESS_GRID, corners, ret)
        
        # --- EXTRAPOLATION LOGIC STARTS HERE ---
        
        # Reshape to 7x7 grid for easy array access [Y, X, coords]
        corners_grid = corners.reshape(CHESS_GRID[1], CHESS_GRID[0], 2)
        
        # 1. Find the Unit Vector (Distance of one square)
        # Vector X: distance from corners[0, 0] to corners[0, 1] (horizontal step)
        vector_x = corners_grid[0, 1] - corners_grid[0, 0]
        # Vector Y: distance from corners[0, 0] to corners[1, 0] (vertical step)
        vector_y = corners_grid[1, 0] - corners_grid[0, 0]

        # 2. Extrapolate the 8x8 Outer Corners by projecting outward
        
        # Top-Left (A8 equivalent): Shift inner corner (0, 0) up and left
        top_left_outer = corners_grid[0, 0] - vector_x - vector_y
        
        # Top-Right (H8 equivalent): Shift inner corner (0, 6) up and right
        top_right_outer = corners_grid[0, 6] + vector_x - vector_y
        
        # Bottom-Right (H1 equivalent): Shift inner corner (6, 6) down and right
        bottom_right_outer = corners_grid[6, 6] + vector_x + vector_y
        
        # Bottom-Left (A1 equivalent): Shift inner corner (6, 0) down and left
        bottom_left_outer = corners_grid[6, 0] - vector_x + vector_y
        
        # 3. Final Source Points for Homography
        source_points_extrapolated = np.float32([
            top_left_outer, 
            top_right_outer, 
            bottom_right_outer, 
            bottom_left_outer
        ])

        # Draw the EXTRAPOLATED points (Yellow circles) for confirmation
        for p in source_points_extrapolated:
            cv2.circle(frame_display, tuple(p.astype(int)), 10, (0, 255, 255), 2)
            
        source_points_final = source_points_extrapolated # Store for saving

    cv2.imshow('Calibration Frame', frame_display)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c') and ret:
        print("✅ 4 Source corners captured! Press S to save the matrix.")
        # Points are already stored in source_points_final if ret was true
        
    elif key == ord('s') and source_points_final is not None:
        # 4. Define the 4 Destination Points (a perfect square)
        dst_points = np.float32([
            [0, 0], 
            [BOARD_SIZE_PX, 0], 
            [BOARD_SIZE_PX, BOARD_SIZE_PX], 
            [0, BOARD_SIZE_PX]
        ])
        
        # 5. Calculate the Homography Matrix (H)
        H, _ = cv2.findHomography(source_points_final, dst_points)
        
        # 6. Save the matrix to a file
        np.save('homography_matrix.npy', H)
        print(f"💾 Homography Matrix saved successfully to homography_matrix.npy.")
        break
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()