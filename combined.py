import cv2
from ultralytics import YOLO

def run_yolo_model(model_path, input_source):
    # Load the YOLO model
    model = YOLO(model_path)

    if input_source == '0':  # Webcam
        cap = cv2.VideoCapture(0)
    elif input_source.lower().endswith(('.mp4', '.avi', '.mov')):  # Video file
        cap = cv2.VideoCapture(input_source)
    else:  # Single image
        image = cv2.imread(input_source)
        results = model(image)
        annotated_image = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_user_input():
    print("Choose input source:")
    print("1. Webcam")
    print("2. Video file")
    print("3. Image file")
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        return '0'  # '0' represents the default webcam
    elif choice == '2':
        return input("Enter the path to the video file: ")
    elif choice == '3':
        return input("Enter the path to the image file: ")
    else:
        print("Invalid choice. Please run the script again.")
        exit()

if __name__ == "__main__":
    model_path = r"C:\Users\sambh\Downloads\best.pt"
    input_source = get_user_input()
    run_yolo_model(model_path, input_source)