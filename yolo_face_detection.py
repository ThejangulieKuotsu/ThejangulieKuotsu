from ultralytics import YOLO

human_model = YOLO('yolov8n.pt') 


def has_human(image_path: str) -> bool:
    try:
        results = human_model(image_path, conf=0.8, verbose=False)

        for result in results:
            # Class 0 is 'person' in COCO dataset
            print(result)
            if any(int(box.cls[0]) == 9 for box in result.boxes):
                return True
        return False
    
    except Exception as e:
        print(f"Error in human detection: {e}")

        return False


def main():
    image = "with_face.png" 
 
    print(f"--- Results for {image} ---")

    print(f"Human:     {has_human(image)}")

if __name__ == "__main__":
    main()
