import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ---------------- SETTINGS ----------------
DATASET_PATH = '/Users/yuvi/Desktop/projects/face recog/dataset'
EMBEDDING_PATH = 'embeddings'
THRESHOLD = 0.60  # Similarity threshold
IMG_SIZE = 160

# ---------------- DEVICE SELECTION ----------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[INFO] Using Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[INFO] Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("[INFO] Using CPU")

# ---------------- MODELS ----------------
print("[INFO] Loading models...")
# Keep MTCNN on CPU for compatibility
mtcnn = MTCNN(image_size=IMG_SIZE, margin=10, keep_all=True, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Ensure embedding dir exists
os.makedirs(EMBEDDING_PATH, exist_ok=True)

# ---------------- BUILD EMBEDDINGS ----------------
def build_embeddings():
    embeddings_dict = {}
    print("[INFO] Building embedding database...")

    for person in tqdm(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        saved_embed_file = os.path.join(EMBEDDING_PATH, f"{person}.pt")
        if os.path.exists(saved_embed_file):
            embeddings_dict[person] = torch.load(saved_embed_file)
            continue

        person_embeddings = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"[WARNING] Failed to open image {img_path}: {e}")
                continue

            # Detect face(s) with MTCNN
            faces = mtcnn(img)

            # Skip if nothing detected
            if faces is None:
                continue

            # Handle single face or batch of faces
            if isinstance(faces, torch.Tensor):
                if faces.ndim == 3:
                    # Single face
                    faces = [faces]
                elif faces.ndim == 4:
                    # Batch of faces
                    faces = list(faces)
                else:
                    print(f"[WARNING] Unexpected face tensor shape: {faces.shape}")
                    continue

            for face in faces:
                try:
                    face = face.to(DEVICE)
                    with torch.no_grad():
                        emb = resnet(face.unsqueeze(0)).squeeze()
                        emb = emb / emb.norm()
                        person_embeddings.append(emb.cpu())
                except Exception as e:
                    print(f"[WARNING] Embedding failed for {img_path}: {e}")

        if person_embeddings:
            mean_embedding = torch.stack(person_embeddings).mean(0)
            torch.save(mean_embedding, saved_embed_file)
            embeddings_dict[person] = mean_embedding
            print(f"[INFO] Saved embedding for {person}")
        else:
            print(f"[WARNING] No valid embeddings found for {person}")

    print("[INFO] Embedding database ready.")
    return embeddings_dict


# ---------------- COSINE SIMILARITY ----------------
def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ---------------- LOAD EMBEDDINGS ----------------
embeddings = build_embeddings()

# ---------------- REAL-TIME FACE RECOGNITION ----------------
cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time face recognition...")

prev_time = time.time()

frame_skip = 2  # Process every 3rd frame
frame_count = 0




while True:
    
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        time.sleep(1)
        continue

    # frame_count += 1
    # if frame_count % frame_skip != 0:
    #      # Just show the frame without detection to keep the window updating
    #      cv2.imshow("Face Recognition", frame)
    #      if cv2.waitKey(1) & 0xFF == ord('q'):
    #          break
    #      continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    boxes, _ = mtcnn.detect(img_pil)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
                face_tensor = (face_tensor - 0.5) / 0.5
            except Exception as e:
                print(f"[WARNING] Face preprocessing failed: {e}")
                continue

            with torch.no_grad():
                embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).squeeze()
                embedding = embedding / embedding.norm()

            best_match = "Unknown"
            max_sim = -1

            for name, db_embed in embeddings.items():
                sim = cosine_sim(embedding.cpu(), db_embed)
                if sim > THRESHOLD and sim > max_sim:
                    max_sim = sim
                    best_match = f"{name} ({sim:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, best_match, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # FPS Display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
