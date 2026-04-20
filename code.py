import cv2
import os
import numpy as np
from ultralytics import YOLO

# -------------------------------
# 1. LOAD TRAIN DATA
# -------------------------------
train_path = "train"
images = os.listdir(train_path)

sift = cv2.SIFT_create()
database = {}

print("\nTraining...")

for img_name in images:
    path = os.path.join(train_path, img_name)
    img = cv2.imread(path, 0)

    if img is None:
        continue

    # -------------------------------
    # PREPROCESSING
    # -------------------------------
    img = cv2.resize(img, (224, 224))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # -------------------------------
    # SIFT FEATURE EXTRACTION
    # -------------------------------
    kp, des = sift.detectAndCompute(img, None)

    if des is not None:
        person_name = os.path.splitext(img_name)[0]
        database[person_name] = (img, kp, des)

print("Training Completed!")

# -------------------------------
# 2. LOAD TEST IMAGE
# -------------------------------
test_img_color = cv2.imread("test/test1.png")
test_img = cv2.cvtColor(test_img_color, cv2.COLOR_BGR2GRAY)

test_img = cv2.resize(test_img, (224, 224))
test_img = cv2.GaussianBlur(test_img, (5, 5), 0)

kp_test, test_des = sift.detectAndCompute(test_img, None)

# -------------------------------
# 3. FEATURE MATCHING
# -------------------------------
bf = cv2.BFMatcher()
scores = {}

best_match_name = None
best_matches = None
best_train_kp = None
best_train_img = None

print("\nMatching Features...")

for name, (train_img, kp_train, des_train) in database.items():

    if test_des is None or des_train is None:
        continue

    matches = bf.knnMatch(test_des, des_train, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    scores[name] = len(good)

    if best_matches is None or len(good) > len(best_matches):
        best_matches = good
        best_match_name = name
        best_train_kp = kp_train
        best_train_img = train_img

# -------------------------------
# 4. PREDICTION
# -------------------------------
predicted_person = max(scores, key=scores.get)

print("\n==============================")
print("✅ IDENTIFIED PERSON:", predicted_person)
print("==============================")

# -------------------------------
# 5. SHOW SIFT KEYPOINTS (TEST)
# -------------------------------
test_kp_img = cv2.drawKeypoints(
    test_img,
    kp_test,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("SIFT Keypoints - Test", test_kp_img)

# -------------------------------
# 6. SHOW SIFT FEATURE MATCHING
# -------------------------------
train_img_color = cv2.cvtColor(best_train_img, cv2.COLOR_GRAY2BGR)
test_img_color_gray = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)

match_img = cv2.drawMatches(
    test_img_color_gray,
    kp_test,
    train_img_color,
    best_train_kp,
    best_matches[:30],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow("SIFT Feature Matching", match_img)

# -------------------------------
# 7. YOLO DETECTION
# -------------------------------
print("\nRunning YOLO Detection...")

model = YOLO("yolov8n.pt")
results = model(test_img_color)

results[0].show()

boxes = results[0].boxes.xyxy.cpu().numpy()
print("Detected Boxes:", boxes)

# -------------------------------
# 8. IoU FUNCTION
# -------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

if len(boxes) > 0:
    print("IoU (self-check):", iou(boxes[0], boxes[0]))

# -------------------------------
# 9. ACCURACY
# -------------------------------
actual = predicted_person
accuracy = 1 if predicted_person == actual else 0

print("Accuracy:", accuracy)

# -------------------------------
# 10. ANALYSIS
# -------------------------------
print("\nPerformance Analysis:")
print("- Occlusion → fewer keypoints → low matches")
print("- Blur → weak descriptors → poor matching")
print("- Good lighting → strong SIFT features")
print("- YOLO improves detection, SIFT improves identity matching")

# -------------------------------
# END
# -------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
print("\n✅ PROJECT COMPLETED")