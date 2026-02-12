

# **Astec PPE Detection – Detailed Technical Work Summary**

---

## **1. Initial Challenges**

From the day we received the Astec videos, we identified that the main limitations were caused by **camera infrastructure**, not the model itself.

### Observed constraints:


* Long camera-to-subject distance
* Sub-optimal camera positioning
* PPE items appear very small
* Lighting inconsistencies
* Partial occlusions

Because of these issues:

* Persons and PPE details were barely visible
* Feature information was lost at capture time

As a result:

* **YOLOv8** could not reliably detect persons
* Multiple image enhancement techniques were attempted
* Pre-processing did not help much since the missing detail cannot be reconstructed

---

## **2. Person Detection Improvement**

To address small/distant person detection:

### Implemented:

### SAHI + YOLOv26

### Why:

* Slice-based inference improves detection of small objects
* Helps detect persons far from the camera

### Result:

* Previously missed persons were detected
* Significant recall improvement

---

## **3. Inference Optimization**

SAHI increased inference time.

### Optimizations applied:

* Frame skipping
* Reduced slices
* Efficient batching

### Outcome:

* Reduced latency
* Near real-time performance

---

## **4. PPE Detection – Initial Approach**

### Pipeline:

```
Person Detection → Crop → LLM/Nova PPE classification
```

### Problems:

* Non-deterministic outputs
* Same input → different predictions
* Unstable for production

### Conclusion:

LLM-based PPE detection is **not reliable for consistent inference**

---

## **5. Astec Model Training (Critical Clarification)**

The current Astec PPE model is **NOT a generalized fine-tuned model**.

### What was done:

* Trained only on limited Astec videos
* Adapted quickly to those scenes
* Allowed intentional overfitting

### Therefore:

> This is an environment-specific PoC model that performs well only on similar scenes.

### Behavior:

* High accuracy on known videos
* Poor generalization on unseen setups

---

## **6. Why More Data Alone Will NOT Fix It**

Even if we collect more videos from the same cameras:

* Same resolution
* Same angles
* Same distance
* Same limitations

Then:

* No new visual information is added
* Model keeps learning same patterns
* Still overfits

### Root cause:

### Hardware + camera setup (not dataset size)

---

## **7. True Factors Affecting Performance (Main Highlight)**

Detection accuracy depends primarily on:

### 🔴 Camera position
### 🔴 Camera resolution
### 🔴 Camera range

### 🔴 Field of view

### 🔴 Lighting

Without improving these:

* PPE remains too small to detect
* Details cannot be learned
* Model performance plateaus

> Hardware constraints dominate accuracy more than training changes.

---

## **8. Practical Accuracy Behavior (Real-world Observation)**

### Important engineering observation:

Accuracy does **not jump directly to 95%**.

Instead, it improves incrementally:

```
75% → 80% → 85% → 90% …
```

### Why:

Each improvement comes from:

* Identifying failure scenarios
* Training specifically on those cases
* Adding preprocessing or logic-based fixes

### Examples:

* Missed small persons → SAHI slicing
* Missed frames → tracking
* Flickering → smoothing
* Hard scenes → targeted retraining

### Key takeaway:

> Accuracy improves by systematically handling edge cases, not by one-time training.

---

## **9. Stability Improvements Implemented**

To improve consistency:

* Tracking
* Box smoothing
* Temporal filtering

### Result:

* Reduced flickering
* Smoother annotations
* More stable PPE predictions

### Detection accuracy:

~90–95% on similar/good-quality scenes

---

## **10. Current Inference Pipeline**

We deployed inference using **NVIDIA Triton Inference Server**.

### Endpoints:

1. `person_ppe_astec` → Overfitted PPE PoC model
2. `person_detection_yolo26m` → General detector

### Workflow:

```
S3 Video → Model → Inference → Annotated video → Stats → Presigned URL
```

---

## **11. Counting Limitations (Important Reality)**

Counting persons accurately is inherently difficult **without controlled entry/exit zones**.

### Observed issues:

* Person leaves frame → re-enters → counted twice
* Occlusions break tracking
* Missed detections
* Frame skipping effects
* No fixed ROI/line crossing

### Even tracking algorithms:

According to official implementations, typically achieve only:

### ~80–85% counting accuracy

### Reason:

Without:

* Entry line
* Exit line
* Defined ROI

Counting becomes ambiguous.

### Current workaround:

Mode-based count

### Provides:

* More stable estimate
* Still not exact
* Cannot guarantee true count

### Key point:

> Without ROI/line-based logic, perfect counting is practically impossible.

---

## **12. Architecture Direction**

 By the Discussion we had with  @Yash Sir and @Suman Sir:

### Move from:

Monolithic FastAPI

### Move to:

Microservices with shared DB

### Goal:

* Scalable
* Modular
* Production-ready
* Spectra 2.0 compatible

---

# ✅ **Current Achievements**

* SAHI + YOLOv26 detection
* Small object handling
* Reduced inference time
* Overfitted Astec PPE PoC model
* ~90–95% detection accuracy (similar scenes)
* Tracking + smoothing
* Triton endpoints
* Annotated video output
* Count stats

---

# ⚠️ **Known Limitations**

* Overfitted model
* Limited generalization
* Counting inaccuracies
* Camera-dependent performance
* Hardware constraints dominate
* No ROI-based counting → unavoidable errors

---

# 🚨 **Final Key Statements**

### 1.

> The primary bottleneck is camera positioning and resolution, and range — not training size.

### 2.

> Accuracy improves incrementally by solving specific failure cases, not by one-time training.

### 3.

> Without ROI/entry–exit logic, counting accuracy will always remain approximate (~80–85%).

