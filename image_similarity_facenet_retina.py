from deepface import DeepFace
import numpy as np
import time

def verify_faces(image1, image2,  threshold=0.70):
    """
    Returns True if similarity >= threshold.
    """
    try:
        t1= time.time()
        result = DeepFace.verify(
            img1_path=image1,
            img2_path=image2,
            model_name="Facenet512",
            detector_backend="retinaface",
            distance_metric="cosine"
        )
        print(time.time()-t1)

        distance = result["distance"]
        similarity = 1 - distance
        is_match = similarity >= threshold
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Is Match: {is_match}")
        return is_match

    except ValueError as e:
        print(f"Face Detection Error: {e}")
        return False

verify_faces("db_photo.png", "old_man.png")  
