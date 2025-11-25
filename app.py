import face_recognition
import cv2
import numpy as np
import os 

# --- Cấu hình ---
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_IMAGES_DIR = "unknown_images"
OUTPUT_IMAGES_DIR = "output_images"
TOLERANCE = 0.6  # Ngưỡng (thấp hơn => nghiêm ngặt hơn). 0.6 là tiêu chuẩn tốt.
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "hog"  # "hog" (nhanh hơn, ít chính xác hơn) hoặc "cnn" (chậm hơn, chính xác hơn, cần GPU)

# --- Hàm tải và mã hóa khuôn mặt đã biết ---
def load_known_faces(known_faces_dir):
    print(f"Đang tải và mã hóa khuôn mặt từ thư mục '{known_faces_dir}'...")
    known_face_encodings = []
    known_face_names = []

    for name in os.listdir(known_faces_dir):
        # Đảm bảo đây là thư mục của một người (ví dụ: obama/)
        # hoặc file ảnh trực tiếp (obama.jpg)
        if os.path.isfile(os.path.join(known_faces_dir, name)) and name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(known_faces_dir, name)
            person_name = os.path.splitext(name)[0] # Lấy tên từ tên file (ví dụ: obama từ obama.jpg)

            print(f"  - Đang xử lý: {person_name}...")
            image = face_recognition.load_image_file(image_path)
            
            face_locations = face_recognition.face_locations(image, model=MODEL)
            if not face_locations:
                print(f"    Không tìm thấy khuôn mặt nào trong ảnh {image_path}. Bỏ qua.")
                continue

            # Chúng ta giả định mỗi ảnh "đã biết" chỉ có một khuôn mặt chính
            encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
            
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)
        elif os.path.isdir(os.path.join(known_faces_dir, name)):
            # Nếu bạn có các thư mục con cho mỗi người (ví dụ: known_faces/obama/pic1.jpg)
            # Điều này phức tạp hơn một chút, chúng ta sẽ giữ cho ví dụ này đơn giản
            print(f"  - Bỏ qua thư mục con '{name}'. Chỉ xử lý file ảnh trực tiếp.")
            pass
    
    print(f"Đã tải {len(known_face_encodings)} khuôn mặt đã biết.")
    return known_face_encodings, known_face_names

# --- Hàm nhận diện khuôn mặt trong ảnh chưa biết ---
# --- Hàm nhận diện khuôn mặt trong ảnh chưa biết ---
def recognize_faces_in_image(image_path, known_face_encodings, known_face_names):
    print(f"\nĐang xử lý ảnh chưa biết: {image_path}...")
    

    # Tải ảnh bằng face_recognition (dùng Pillow) để xử lý (ảnh RGB)
    rgb_image = face_recognition.load_image_file(image_path)
    
    # Tải ảnh bằng OpenCV (BGR) CHỈ để dùng cho việc vẽ và lưu
    image_to_draw = cv2.imread(image_path)
    if image_to_draw is None:
        print(f"Không thể tải ảnh bằng OpenCV: {image_path}. Bỏ qua.")
        return
    

    # Tìm tất cả khuôn mặt và encoding của chúng trong ảnh RGB
    # (Giờ đây chúng ta sử dụng rgb_image, không phải ảnh từ cv2)
    face_locations = face_recognition.face_locations(rgb_image, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)

    print(f"  Tìm thấy {len(face_locations)} khuôn mặt trong ảnh này.")

    # Lặp qua từng khuôn mặt tìm thấy
    for face_location, face_encoding in zip(face_locations, face_encodings):
        
        # So sánh khuôn mặt chưa biết với danh sách các khuôn mặt đã biết
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)

        name = "Khong biet"
        
        # Nếu tìm thấy một kết quả khớp, tìm người gần nhất
        if True in matches:
            # Tìm khoảng cách từ khuôn mặt chưa biết đến tất cả khuôn mặt đã biết
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # Lấy index của khuôn mặt có khoảng cách nhỏ nhất (nghĩa là giống nhất)
            best_match_index = np.argmin(face_distances)
            
            # Chỉ gán tên nếu khuôn mặt gần nhất cũng là một match hợp lệ
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Lấy tọa độ khuôn mặt
        top, right, bottom, left = face_location

        # Vẽ hộp và tên lên ảnh (Vẽ lên ảnh của OpenCV)
        cv2.rectangle(image_to_draw, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)
        cv2.putText(image_to_draw, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), FONT_THICKNESS)
        print(f"  - Khuôn mặt ở ({left},{top}) được nhận diện là: {name}")

    # Lưu ảnh kết quả (Lưu ảnh của OpenCV)
    output_filename = "recognized_" + os.path.basename(image_path)
    output_path = os.path.join(OUTPUT_IMAGES_DIR, output_filename)
    cv2.imwrite(output_path, image_to_draw)
    print(f"  Đã lưu ảnh kết quả tới: {output_path}")

# --- Chương trình chính ---
if __name__ == "__main__":
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)

    # Tải và mã hóa tất cả khuôn mặt đã biết
    known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)

    # Xử lý từng ảnh trong thư mục unknown_images
    for filename in os.listdir(UNKNOWN_IMAGES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(UNKNOWN_IMAGES_DIR, filename)
            recognize_faces_in_image(image_path, known_face_encodings, known_face_names)

    print("\nQuá trình nhận diện hoàn tất!")
    print(f"Kiểm tra thư mục '{OUTPUT_IMAGES_DIR}' để xem kết quả.")