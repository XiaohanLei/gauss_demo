import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def create_dataset_folder(base_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dataset_path = os.path.join(base_path, f"dataset_{timestamp}")
    os.makedirs(dataset_path, exist_ok=True)
    return dataset_path

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable both color and depth streams
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create dataset folder
    dataset_path = create_dataset_folder("datasets")
    frame_count = 0

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                # Save color and depth images
                color_filename = os.path.join(dataset_path, f"color_{frame_count:04d}.png")
                depth_filename = os.path.join(dataset_path, f"depth_{frame_count:04d}.png")
                
                cv2.imwrite(color_filename, color_image)
                cv2.imwrite(depth_filename, depth_image)
                
                print(f"Saved frame {frame_count}")
                frame_count += 1

    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    main()
