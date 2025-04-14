import cv2
import numpy as np
import time
import os
import argparse

# Function to find the nearest multiple of a size to fit with window_size
def find_nearest_multiple(size, factor, max_deviation=3):
    """
    Find the nearest size that is a multiple of factor.
    Args:
        size (int): Original size.
        factor (int): Factor (window size).
        max_deviation (int): Maximum allowed percentage deviation from original size.
    Returns:
        int: New size that is the nearest multiple of factor to size.
    """
    new_size = round(size / factor) * factor
    
    # Kiểm tra xem thay đổi kích thước có vượt quá giới hạn cho phép không
    deviation_percent = abs(new_size - size) / size * 100
    if deviation_percent > max_deviation:
        # Nếu vượt quá giới hạn, chọn multiple gần nhất
        lower_multiple = (size // factor) * factor
        upper_multiple = lower_multiple + factor
        
        if abs(lower_multiple - size) <= abs(upper_multiple - size):
            new_size = lower_multiple
        else:
            new_size = upper_multiple
    
    return new_size

# Function to process a frame with NumPy optimization
def process_frame(frame, window_size=5, maintain_size=True):
    """
    Process a frame with circle effect.
    Args:
        frame (numpy.ndarray): Input frame.
        window_size (int): Window size.
        maintain_size (bool): Whether to maintain original frame size.
    Returns:
        numpy.ndarray: Processed frame.
    """
    original_shape = frame.shape
    
    # Convert frame to grayscale if needed
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    original_height, original_width = gray_frame.shape

    # Calculate new dimensions as multiples of window_size
    new_width = find_nearest_multiple(original_width, window_size)
    new_height = find_nearest_multiple(original_height, window_size)

    # Resize frame if needed
    if (new_width, new_height) != (original_width, original_height):
        gray_frame = cv2.resize(gray_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create output array with initial black (0) values
    output_array = np.zeros_like(gray_frame, dtype=np.uint8)

    # Calculate number of windows
    num_windows_y = new_height // window_size
    num_windows_x = new_width // window_size
    
    # Optimization: Calculate distance matrix once
    y_indices, x_indices = np.indices((window_size, window_size))
    center_y, center_x = window_size // 2, window_size // 2
    dist_squared = (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2
    
    # Optimization: Calculate all average values at once using resize
    small_img = cv2.resize(gray_frame, (num_windows_x, num_windows_y), interpolation=cv2.INTER_AREA)
    
    # Loop through all windows
    for i in range(num_windows_y):
        for j in range(num_windows_x):
            # Get average value from resized small_img
            WAvr = small_img[i, j]
            
            # Calculate radius and threshold
            max_radius = window_size / 2
            radius = (WAvr / 255) * max_radius
            threshold = radius ** 2
            
            # Create mask for circle using NumPy
            mask = dist_squared <= threshold
            
            # Draw circle at corresponding position in output array
            output_array[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size][mask] = 255

    # Convert array to 3-channel frame (for saving color video)
    output_frame = cv2.cvtColor(output_array, cv2.COLOR_GRAY2BGR)
    
    # Resize back to original dimensions if maintain_size is True
    if maintain_size and (new_width, new_height) != (original_width, original_height):
        output_frame = cv2.resize(output_frame, (original_width, original_height), interpolation=cv2.INTER_AREA)
    
    # Ensure output frame has same number of channels as input
    if len(original_shape) == 3 and original_shape[2] == output_frame.shape[2]:
        return output_frame
    elif len(original_shape) == 2:
        return cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
    
    return output_frame


# Function to process image
def create_artistic_image(input_path, output_path=None, window_size=5, save=True, show=False):
    """
    Convert an image to an artistic image with white circles on a black background.
    Args:
        input_path (str): Path to input image.
        output_path (str, optional): Path to save output image.
        window_size (int): Window size (default is 5).
        save (bool): Whether to save the output image.
        show (bool): Whether to display the output image.
    Returns:
        numpy.ndarray: Processed image.
    """
    # Check valid window_size
    if window_size <= 0 or not isinstance(window_size, int):
        raise ValueError("Window size must be a positive integer.")

    # Read image using OpenCV
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot read image from {input_path}")
    
    # Process image
    print(f"Processing image '{input_path}' with window_size={window_size}...")
    start_time = time.time()
    output_image = process_frame(img, window_size)
    elapsed_time = time.time() - start_time
    print(f"Image processing completed in {elapsed_time:.2f} seconds")
    
    # Display image if needed
    if show:
        cv2.imshow('Original Image', img)
        cv2.imshow('Artistic Image', output_image)
        print("Press any key to close display window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save image if needed
    if save and output_path:
        cv2.imwrite(output_path, output_image)
        print(f"Artistic image has been saved at: {output_path}")
    
    return output_image

# Function to process video
def process_video(input_path, output_path=None, window_size=5, batch_size=10, save=True, show=False):
    """
    Process video with circle effect.
    Args:
        input_path (str): Path to input video.
        output_path (str, optional): Path to save output video.
        window_size (int): Window size (default is 5).
        batch_size (int): Number of frames to process simultaneously (default is 10).
        save (bool): Whether to save the output video.
        show (bool): Whether to display the processing.
    """
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video from {input_path}")
        return

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Detect video orientation
    is_portrait = original_height > original_width
    if is_portrait:
        print(f"Detected portrait video: {original_width}x{original_height}")
    
    # Check if video is larger than Full HD (1920x1080)
    need_resize = original_width > 1920 or original_height > 1080
    if need_resize:
        # Calculate ratio to maintain aspect ratio
        resize_ratio = min(1920 / original_width, 1080 / original_height)
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)
        print(f"Resizing video from {original_width}x{original_height} to {new_width}x{new_height}")
    else:
        new_width = original_width
        new_height = original_height

    # Decide whether to process in batches
    use_batch = fps > 5
    if use_batch:
        actual_batch_size = min(batch_size, max(1, int(fps // 2)))
        print(f"Video has FPS = {fps}, using batch_size = {actual_batch_size}")
    else:
        actual_batch_size = 1
    
    # Process first frame to get the exact output size
    ret, first_frame = cap.read()
    if not ret:
        print("Cannot read frames from the video")
        return
    
    if need_resize:
        first_frame = cv2.resize(first_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Process first frame to determine exact output size
    first_processed_frame = process_frame(first_frame, window_size)
    output_height, output_width = first_processed_frame.shape[:2]
    
    print(f"Input frame size: {new_width}x{new_height}")
    print(f"Output frame size: {output_width}x{output_height}")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create video writer to save output video if needed
    if save and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        if not out.isOpened():
            print(f"Warning: Could not initialize VideoWriter with size {output_width}x{output_height}")
            print("Trying with swapped dimensions...")
            # Try with swapped dimensions as a fallback
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_height, output_width))
            if not out.isOpened():
                print("Error: Failed to create output video file. Continuing without saving...")
                save = False
            else:
                print(f"Successfully created VideoWriter with swapped dimensions: {output_height}x{output_width}")

    # Variables to track progress
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()
    last_update_time = start_time
    frames_since_last_update = 0
    
    print(f"Processing video '{input_path}' with window_size={window_size}...")
    
    # Use read_batch to read multiple frames at once
    def read_batch(cap, size, need_resize, new_width, new_height):
        frames = []
        for _ in range(size):
            ret, frame = cap.read()
            if not ret:
                break
            if need_resize:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        return frames
    
    # Process frames in batches
    while True:
        # Read a batch of frames
        batch_frames = read_batch(cap, actual_batch_size if use_batch else 1, need_resize, new_width, new_height)
        if not batch_frames:
            break
        
        # Process each frame in the batch
        for i, frame in enumerate(batch_frames):
            # Process frame
            processed_frame = process_frame(frame, window_size)
            
            # Display original and processed frames (only display first frame in batch)
            if show and i == 0:  # Only display first frame of each batch to avoid slowing down processing
                cv2.imshow('Original Video', frame)
                cv2.imshow('Artistic Video', processed_frame)
                # Press 'q' to exit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame to output video
            if save and output_path and 'out' in locals():
                # Verify frame dimensions match with what VideoWriter expects
                if processed_frame.shape[1] != output_width or processed_frame.shape[0] != output_height:
                    processed_frame = cv2.resize(processed_frame, (output_width, output_height))
                try:
                    out.write(processed_frame)
                except Exception as e:
                    print(f"Error writing frame: {e}")
                    print(f"Frame shape: {processed_frame.shape}, Expected: {output_width}x{output_height}")
                    save = False
            
            frame_count += 1
            frames_since_last_update += 1
            
            # Update progress in real-time
            current_time = time.time()
            if current_time - last_update_time >= 1.0:  # Update every second
                elapsed_time = current_time - start_time
                fps_processing = frames_since_last_update / (current_time - last_update_time)
                percent_done = frame_count / total_frames * 100 if total_frames > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({percent_done:.1f}%) - Processing FPS: {fps_processing:.2f}")
                last_update_time = current_time
                frames_since_last_update = 0
        
        # If 'q' was pressed in the above loop, exit the main loop
        if show and (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # Release resources
    cap.release()
    if save and output_path and 'out' in locals():
        out.release()
    if show:
        cv2.destroyAllWindows()
    
    # Display result information
    elapsed_time = time.time() - start_time
    print(f"Finished processing {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {frame_count/elapsed_time:.2f} FPS")
    if save and output_path:
        print(f"Video has been saved at: {output_path}")
        
# Function to process webcam
def process_webcam(window_size=5, output_path=None, save=True, show=True):
    """
    Process webcam with circle effect.
    Args:
        window_size (int): Window size (default is 5).
        output_path (str, optional): Path to save output video.
        save (bool): Whether to save the output video.
        show (bool): Whether to display the processing.
    """
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    # Get webcam information
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Check if webcam is larger than Full HD (1920x1080)
    need_resize = width > 1920 or height > 1080
    if need_resize:
        # Calculate ratio to maintain aspect ratio
        resize_ratio = min(1920 / width, 1080 / height)
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)
        print(f"Resizing webcam from {width}x{height} to {new_width}x{new_height}")
    else:
        new_width = width
        new_height = height

    # Create video writer if output path is provided and save=True
    if save and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (new_width, new_height))

    # Variables to measure FPS
    frame_count = 0
    start_time = time.time()
    fps_display_time = start_time
    
    print(f"Processing webcam with window_size={window_size}...")
    print("Press 'q' to stop.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame if needed
        if need_resize:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        # Process frame
        processed_frame = process_frame(frame, window_size)

        # Display original and processed frames
        if show:
            cv2.imshow('Original Webcam', frame)
            cv2.imshow('Artistic Webcam', processed_frame)

        # Write frame to output video if needed
        if save and output_path:
            out.write(processed_frame)
            
        # Count frames and display FPS every second
        frame_count += 1
        current_time = time.time()
        if current_time - fps_display_time >= 1.0:
            fps = frame_count / (current_time - fps_display_time)
            print(f"Processing FPS: {fps:.2f}")
            frame_count = 0
            fps_display_time = current_time

        # Press 'q' to exit
        if show and (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # Release resources
    cap.release()
    if save and output_path:
        out.release()
    if show:
        cv2.destroyAllWindows()
    
    if save and output_path:
        elapsed_time = time.time() - start_time
        print(f"Video has been saved at: {output_path}")
        print(f"Recording time: {elapsed_time:.2f} seconds")

def main():
    """
    Main function to process command line arguments and call corresponding functions.
    """
    # Create parser for CLI
    parser = argparse.ArgumentParser(
        description="Program to create artistic effects using circles for images and videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common parameters
    parser.add_argument("input", help="Path to input file (image or video) or 'webcam' to use camera")
    parser.add_argument("--output", "-o", help="Path to save the result. If not provided, will be automatically created")
    parser.add_argument("--window-size", "-w", type=int, default=5, 
                        help="Window size (default is 5)")
    parser.add_argument("--save", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True,
                        help="Whether to save the result (default is True)")
    parser.add_argument("--show", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Whether to display the processing (default is False)")
    
    # Parameters for video
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Number of frames to process simultaneously for video (default is 10)")
    
    args = parser.parse_args()
    
    # Automatically create output filename if not provided
    if args.output is None:
        if args.input.lower() == 'webcam':
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            args.output = f"webcam_art_{timestamp}.mp4"
        else:
            input_name, input_ext = os.path.splitext(args.input)
            args.output = f"{input_name}_art{input_ext}"
    
    # Check input type and process accordingly
    if args.input.lower() == 'webcam':
        process_webcam(
            window_size=args.window_size,
            output_path=args.output if args.save else None,
            save=args.save,
            show=True  # Display is always enabled for webcam
        )
    else:
        # Check if input is an image or video
        input_ext = os.path.splitext(args.input)[1].lower()
        is_image = input_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if is_image:
            create_artistic_image(
                input_path=args.input,
                output_path=args.output if args.save else None,
                window_size=args.window_size,
                save=args.save,
                show=args.show
            )
        else:
            process_video(
                input_path=args.input,
                output_path=args.output if args.save else None,
                window_size=args.window_size,
                batch_size=args.batch_size,
                save=args.save,
                show=args.show
            )

if __name__ == "__main__":
    main()