import cv2
import numpy as np
import time
import os
import argparse

# Hàm tìm bội số gần nhất của kích thước để phù hợp với window_size
def find_nearest_multiple(size, factor):
    """
    Tìm kích thước gần nhất là bội của factor.
    Args:
        size (int): Kích thước ban đầu.
        factor (int): Hệ số (kích thước window).
    Returns:
        int: Kích thước mới là bội của factor gần nhất với size.
    """
    return round(size / factor) * factor

# Hàm xử lý một khung hình với tối ưu NumPy
def process_frame(frame, window_size=5):
    """
    Xử lý một khung hình với hiệu ứng hình tròn.
    Args:
        frame (numpy.ndarray): Khung hình đầu vào.
        window_size (int): Kích thước window.
    Returns:
        numpy.ndarray: Khung hình đã xử lý.
    """
    # Chuyển frame thành grayscale nếu cần
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    original_height, original_width = gray_frame.shape

    # Tính kích thước mới là bội của window_size
    new_width = find_nearest_multiple(original_width, window_size)
    new_height = find_nearest_multiple(original_height, window_size)

    # Resize frame nếu cần
    if (new_width, new_height) != (original_width, original_height):
        gray_frame = cv2.resize(gray_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Tạo mảng đầu ra với giá trị ban đầu là đen (0)
    output_array = np.zeros_like(gray_frame, dtype=np.uint8)

    # Tính số lượng window
    num_windows_y = new_height // window_size
    num_windows_x = new_width // window_size
    
    # Tối ưu hóa: Tính toán ma trận khoảng cách một lần
    y_indices, x_indices = np.indices((window_size, window_size))
    center_y, center_x = window_size // 2, window_size // 2
    dist_squared = (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2
    
    # Tối ưu hóa: Tính toán tất cả giá trị trung bình cùng lúc bằng resize
    small_img = cv2.resize(gray_frame, (num_windows_x, num_windows_y), interpolation=cv2.INTER_AREA)
    
    # Lặp qua tất cả windows
    for i in range(num_windows_y):
        for j in range(num_windows_x):
            # Lấy giá trị trung bình từ small_img đã resize
            WAvr = small_img[i, j]
            
            # Tính bán kính và ngưỡng
            max_radius = window_size / 2
            radius = (WAvr / 255) * max_radius
            threshold = radius ** 2
            
            # Tạo mặt nạ cho hình tròn bằng NumPy
            mask = dist_squared <= threshold
            
            # Vẽ hình tròn vào vị trí tương ứng trong mảng đầu ra
            output_array[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size][mask] = 255

    # Chuyển mảng thành frame 3 kênh (để lưu video màu)
    output_frame = cv2.cvtColor(output_array, cv2.COLOR_GRAY2BGR)
    return output_frame

# Hàm xử lý ảnh
def create_artistic_image(input_path, output_path=None, window_size=5, save=True, show=False):
    """
    Chuyển đổi ảnh thành ảnh nghệ thuật với các hình tròn trắng trên nền đen.
    Args:
        input_path (str): Đường dẫn đến ảnh đầu vào.
        output_path (str, optional): Đường dẫn để lưu ảnh đầu ra.
        window_size (int): Kích thước của window (mặc định là 5).
        save (bool): Có lưu ảnh đầu ra hay không.
        show (bool): Có hiển thị ảnh đầu ra hay không.
    Returns:
        numpy.ndarray: Ảnh đã xử lý.
    """
    # Kiểm tra window_size hợp lệ
    if window_size <= 0 or not isinstance(window_size, int):
        raise ValueError("Kích thước window phải là số nguyên dương.")

    # Đọc ảnh bằng OpenCV
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {input_path}")
    
    # Xử lý ảnh
    print(f"Đang xử lý ảnh '{input_path}' với window_size={window_size}...")
    start_time = time.time()
    output_image = process_frame(img, window_size)
    elapsed_time = time.time() - start_time
    print(f"Đã xử lý xong ảnh trong {elapsed_time:.2f} giây")
    
    # Hiển thị ảnh nếu cần
    if show:
        cv2.imshow('Original Image', img)
        cv2.imshow('Artistic Image', output_image)
        print("Nhấn phím bất kỳ để đóng cửa sổ hiển thị...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Lưu ảnh nếu cần
    if save and output_path:
        cv2.imwrite(output_path, output_image)
        print(f"Ảnh nghệ thuật đã được lưu tại: {output_path}")
    
    return output_image

# Hàm xử lý video
def process_video(input_path, output_path=None, window_size=5, batch_size=10, save=True, show=False):
    """
    Xử lý video với hiệu ứng hình tròn.
    Args:
        input_path (str): Đường dẫn đến video đầu vào.
        output_path (str, optional): Đường dẫn để lưu video đầu ra.
        window_size (int): Kích thước của window (mặc định là 5).
        batch_size (int): Số lượng frame xử lý cùng lúc (mặc định là 10).
        save (bool): Có lưu video đầu ra hay không.
        show (bool): Có hiển thị quá trình xử lý hay không.
    """
    # Mở video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Không thể mở video từ {input_path}")
        return

    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Kiểm tra nếu video lớn hơn Full HD (1920x1080)
    need_resize = original_width > 1920 or original_height > 1080
    if need_resize:
        # Tính toán tỷ lệ để giữ nguyên tỷ lệ khung hình
        resize_ratio = min(1920 / original_width, 1080 / original_height)
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)
        print(f"Resizing video từ {original_width}x{original_height} thành {new_width}x{new_height}")
    else:
        new_width = original_width
        new_height = original_height

    # Quyết định có nên xử lý batch hay không
    use_batch = fps > 5
    if use_batch:
        actual_batch_size = min(batch_size, max(1, int(fps // 2)))
        print(f"Video có FPS = {fps}, sử dụng batch_size = {actual_batch_size}")
    else:
        actual_batch_size = 1
    
    # Tạo video writer để lưu video đầu ra nếu cần
    if save and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    # Biến để theo dõi tiến trình
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()
    last_update_time = start_time
    frames_since_last_update = 0
    
    print(f"Đang xử lý video '{input_path}' với window_size={window_size}...")
    
    # Sử dụng read_batch để đọc nhiều frame một lúc
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
    
    # Xử lý các frame theo batch
    while True:
        # Đọc một batch các frame
        batch_frames = read_batch(cap, actual_batch_size if use_batch else 1, need_resize, new_width, new_height)
        if not batch_frames:
            break
        
        # Xử lý từng frame trong batch
        for i, frame in enumerate(batch_frames):
            # Xử lý frame
            processed_frame = process_frame(frame, window_size)
            
            # Hiển thị frame gốc và frame đã xử lý (chỉ hiển thị frame đầu tiên trong batch)
            if show and i == 0:  # Chỉ hiển thị frame đầu tiên của mỗi batch để không làm chậm xử lý
                cv2.imshow('Original Video', frame)
                cv2.imshow('Artistic Video', processed_frame)
                # Nhấn 'q' để thoát sớm
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Ghi frame vào video đầu ra
            if save and output_path:
                out.write(processed_frame)
            
            frame_count += 1
            frames_since_last_update += 1
            
            # Cập nhật tiến trình theo khoảng thời gian thực
            current_time = time.time()
            if current_time - last_update_time >= 1.0:  # Cập nhật mỗi giây
                elapsed_time = current_time - start_time
                fps_processing = frames_since_last_update / (current_time - last_update_time)
                percent_done = frame_count / total_frames * 100 if total_frames > 0 else 0
                print(f"Đã xử lý {frame_count}/{total_frames} frames ({percent_done:.1f}%) - FPS xử lý: {fps_processing:.2f}")
                last_update_time = current_time
                frames_since_last_update = 0
        
        # Nếu đã nhấn 'q' trong vòng lặp trên, thoát vòng lặp chính
        if show and (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # Giải phóng tài nguyên
    cap.release()
    if save and output_path:
        out.release()
    if show:
        cv2.destroyAllWindows()
    
    # Hiển thị thông tin kết quả
    elapsed_time = time.time() - start_time
    print(f"Đã xử lý xong {frame_count} frames trong {elapsed_time:.2f} giây")
    print(f"Tốc độ xử lý trung bình: {frame_count/elapsed_time:.2f} FPS")
    if save and output_path:
        print(f"Video đã được lưu tại: {output_path}")

# Hàm xử lý webcam
def process_webcam(window_size=5, output_path=None, save=True, show=True):
    """
    Xử lý webcam với hiệu ứng hình tròn.
    Args:
        window_size (int): Kích thước của window (mặc định là 5).
        output_path (str, optional): Đường dẫn để lưu video đầu ra.
        save (bool): Có lưu video đầu ra hay không.
        show (bool): Có hiển thị quá trình xử lý hay không.
    """
    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    # Lấy thông tin webcam
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Kiểm tra nếu webcam lớn hơn Full HD (1920x1080)
    need_resize = width > 1920 or height > 1080
    if need_resize:
        # Tính toán tỷ lệ để giữ nguyên tỷ lệ khung hình
        resize_ratio = min(1920 / width, 1080 / height)
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)
        print(f"Resizing webcam từ {width}x{height} thành {new_width}x{new_height}")
    else:
        new_width = width
        new_height = height

    # Tạo video writer nếu có đường dẫn đầu ra và save=True
    if save and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (new_width, new_height))

    # Biến để đo FPS
    frame_count = 0
    start_time = time.time()
    fps_display_time = start_time
    
    print(f"Đang xử lý webcam với window_size={window_size}...")
    print("Nhấn 'q' để dừng.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame nếu cần
        if need_resize:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        # Xử lý frame
        processed_frame = process_frame(frame, window_size)

        # Hiển thị frame gốc và frame đã xử lý
        if show:
            cv2.imshow('Original Webcam', frame)
            cv2.imshow('Artistic Webcam', processed_frame)

        # Ghi frame vào video đầu ra nếu có
        if save and output_path:
            out.write(processed_frame)
            
        # Đếm frame và hiển thị FPS mỗi giây
        frame_count += 1
        current_time = time.time()
        if current_time - fps_display_time >= 1.0:
            fps = frame_count / (current_time - fps_display_time)
            print(f"FPS xử lý: {fps:.2f}")
            frame_count = 0
            fps_display_time = current_time

        # Nhấn 'q' để thoát
        if show and (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # Giải phóng tài nguyên
    cap.release()
    if save and output_path:
        out.release()
    if show:
        cv2.destroyAllWindows()
    
    if save and output_path:
        elapsed_time = time.time() - start_time
        print(f"Video đã được lưu tại: {output_path}")
        print(f"Thời gian ghi: {elapsed_time:.2f} giây")

def main():
    """
    Hàm chính xử lý tham số dòng lệnh và gọi các hàm tương ứng.
    """
    # Tạo parser cho CLI
    parser = argparse.ArgumentParser(
        description="Chương trình tạo hiệu ứng nghệ thuật bằng hình tròn cho ảnh và video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Các tham số chung
    parser.add_argument("input", help="Đường dẫn đến file đầu vào (ảnh hoặc video) hoặc 'webcam' để sử dụng camera")
    parser.add_argument("--output", "-o", help="Đường dẫn để lưu kết quả. Nếu không được cung cấp, sẽ tự động tạo")
    parser.add_argument("--window-size", "-w", type=int, default=5, 
                        help="Kích thước của window (mặc định là 5)")
    parser.add_argument("--save", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True,
                        help="Có lưu kết quả hay không (mặc định là True)")
    parser.add_argument("--show", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Có hiển thị quá trình xử lý hay không (mặc định là False)")
    
    # Các tham số cho video
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Số lượng frame xử lý cùng lúc cho video (mặc định là 10)")
    
    args = parser.parse_args()
    
    # Tự động tạo tên file đầu ra nếu không được cung cấp
    if args.output is None:
        if args.input.lower() == 'webcam':
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            args.output = f"webcam_art_{timestamp}.mp4"
        else:
            input_name, input_ext = os.path.splitext(args.input)
            args.output = f"{input_name}_art{input_ext}"
    
    # Kiểm tra loại đầu vào và xử lý tương ứng
    if args.input.lower() == 'webcam':
        process_webcam(
            window_size=args.window_size,
            output_path=args.output if args.save else None,
            save=args.save,
            show=True  # Hiển thị luôn được bật cho webcam
        )
    else:
        # Kiểm tra xem đầu vào là ảnh hay video
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