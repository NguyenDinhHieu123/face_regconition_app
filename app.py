import face_recognition
import cv2
import numpy as np
import os 

# --- Cấu hình ---
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_IMAGES_DIR = "unknown_images"
OUTPUT_IMAGES_DIR = "output_images"
MODEL = "hog" # có thể sử dụng cnn để tăng hiệu quả nhưng sẽ mất thời gian hơn và yêu cầu gpu
# Hàm nạp dữ liệu ảnh mẫu và mã hóa khuôn mặt
def load_known_faces(known_faces_dir):
    print(f"Đang tải dữ liệu mẫu...")
    known_encodings = []
    known_names = []

    for name in os.listdir(known_faces_dir):
        image_path = os.path.join(known_faces_dir, name)
        if os.path.isfile(image_path) and name.lower().endswith(('.png', '.jpg', '.jpeg')):
            person_name = os.path.splitext(name)[0]
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"  + Đã nạp: {person_name}")
            else:
                print(f"  ! Cảnh báo: Không thấy mặt trong ảnh {name}")
    
    return known_encodings, known_names
# Hàm xử lý ảnh đầu vào và so khớp khuôn mặt
def recognize_faces_in_image(image_path, known_encodings, known_names):
    print(f"\nĐang xử lý: {image_path}...")
    
    # 1. Đọc ảnh 
    rgb_image = face_recognition.load_image_file(image_path)
    
    # 2. Tạo ảnh cho OpenCV từ ảnh gốc (Chuyển RGB -> BGR)
    image_to_draw = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 3. Tìm vị trí và mã hóa
    face_locations = face_recognition.face_locations(rgb_image, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, 0.5)
        name = "Khong biet"

        if True in matches:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Vẽ hình
        cv2.rectangle(image_to_draw, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_to_draw, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"- Khuôn mặt ở {left,top} được nhận diện là : {name}")
    # Lưu ảnh
    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)  

    output_filename = "recognized_" + os.path.basename(image_path)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, output_filename), image_to_draw)
    print(f"  -> Xong.")
# Main
if __name__ == "__main__":
    encodings, names = load_known_faces(KNOWN_FACES_DIR)
    
    if not os.path.exists(UNKNOWN_IMAGES_DIR):
        print(f"Lỗi: Không tìm thấy thư mục {UNKNOWN_IMAGES_DIR}")
    else:
        for filename in os.listdir(UNKNOWN_IMAGES_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                recognize_faces_in_image(os.path.join(UNKNOWN_IMAGES_DIR, filename), encodings, names)