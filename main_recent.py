import os
import re
import io
import time
import shutil
import subprocess
import tempfile
import logging
import asyncio
import functools
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque, Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse, unquote

import cv2
import numpy as np
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import tritonclient.http as httpclient
from tritonclient.http import InferInput, InferRequestedOutput

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# APP
# -------------------------
app = FastAPI(title="YOLO Video Processing API", version="1.0.0")

# -------------------------
# ASYNC CONCURRENCY CONTROLS
# -------------------------
MAX_CONCURRENT_REQUESTS = 50
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

async def run_blocking(func, *args, **kwargs):
    async with REQUEST_SEMAPHORE:
        loop = asyncio.get_running_loop()
        bound = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(EXECUTOR, bound)

# -------------------------
# AWS / S3 CONFIG
# -------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "spectra-manifacturing-usecase")
S3_OUTPUT_BUCKET = os.getenv("S3_OUTPUT_BUCKET", "spectra-manifacturing-usecase")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PRESIGNED_EXPIRY = 7 * 24 * 3600  # 7 days
MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB
FFMPEG_CONVERSION_TIMEOUT = 600  # seconds
PROGRESS_LOG_INTERVAL = 50  # frames

s3_client = boto3.client("s3", region_name=AWS_REGION)

# -------------------------
# TRITON CONFIG
# -------------------------
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
TARGET_SIZE = 640
PAD_VALUE = 114

# === PPE thresholds (from Triton_PPE.py) ===
PERSON_CONF_THRES = 0.16
PPE_CONF_THRES = 0.10
PPE_TRACK_HISTORY = 20
PPE_VOTE_THRESHOLD = 0.4
PPE_MISSING_TOLERANCE = 8
PPE_IOU_MATCH_THRESHOLD = 0.3
PPE_BOX_SMOOTH_ALPHA = 0.7
PERSON_NMS_IOU = 0.3
PPE_NMS_IOU = 0.25

PPE_CLASS_MAP = {
    0: "glasses", 1: "gloves", 2: "helmet",
    3: "no-glasses", 4: "no-gloves", 5: "no-helmet",
    6: "no-shoes", 7: "no-vest", 8: "shoes",
    9: "vest", 10: "person"
}
PPE_PRESENT_CLASSES = {"glasses", "gloves", "helmet", "shoes", "vest"}
PPE_ABSENT_CLASSES = {"no-glasses", "no-gloves", "no-helmet", "no-shoes", "no-vest"}
PPE_ITEMS = ["helmet", "glasses", "vest", "gloves", "shoes"]
DISPLAY_NAME = {
    "helmet": "Hardhat", "glasses": "Goggles", "vest": "Vest",
    "gloves": "Gloves", "shoes": "Shoes"
}

# === Traffic thresholds (from Triton_Traffic.py) ===
TRAFFIC_CONF_THRES = 0.3
TRAFFIC_TRACK_HISTORY = 100
TRAFFIC_MISSING_TOLERANCE = 75
TRAFFIC_IOU_MATCH_THRESHOLD = 0.3
TRAFFIC_BOX_SMOOTH_ALPHA = 0.7

TRAFFIC_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench"
}
TRAFFIC_COLORS = {
    "person": (255, 0, 0), "bicycle": (0, 255, 255), "car": (0, 255, 0),
    "motorcycle": (255, 0, 255), "bus": (0, 165, 255), "truck": (0, 128, 255),
    "traffic light": (0, 0, 255), "stop sign": (0, 0, 200),
    "fire hydrant": (255, 255, 0), "parking meter": (180, 180, 180),
    "train": (255, 100, 0), "airplane": (200, 200, 0),
    "boat": (255, 200, 100), "bench": (128, 128, 128)
}


# -------------------------
# REQUEST / RESPONSE MODELS
# -------------------------
class ProcessVideoRequest(BaseModel):
    video_uri: str
    model_id: str  # "person_detection_yolo26" or "person_ppe_astec"
    frame_skip: int = 0  # 0 = no skip, 1 = skip every other, etc.


# =========================================================================
# S3 HELPERS
# =========================================================================
def _parse_s3_uri(uri: str):
    """Parse s3://bucket/key or presigned URL into (bucket, key)."""
    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        return parsed.netloc, parsed.path.lstrip("/")
    # presigned URL – extract bucket/key from path
    parsed = urlparse(uri)
    # virtual-hosted style: bucket.s3.region.amazonaws.com/key
    host = parsed.hostname or ""
    if ".s3." in host or host.endswith(".s3.amazonaws.com"):
        bucket = host.split(".s3")[0]
        key = unquote(parsed.path.lstrip("/"))
        return bucket, key
    # path style: s3.amazonaws.com/bucket/key
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) == 2:
        return parts[0], unquote(parts[1])
    raise ValueError(f"Cannot parse S3 URI: {uri}")


def _generate_presigned_url(bucket: str, key: str) -> str:
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=PRESIGNED_EXPIRY,
    )


def _multipart_upload(bucket: str, key: str, file_obj, content_type: str = "video/mp4"):
    """Upload using S3 multipart for large files."""
    mpu = s3_client.create_multipart_upload(Bucket=bucket, Key=key, ContentType=content_type)
    upload_id = mpu["UploadId"]
    parts = []
    part_number = 1

    try:
        while True:
            chunk = file_obj.read(MULTIPART_CHUNK_SIZE)
            if not chunk:
                break
            resp = s3_client.upload_part(
                Bucket=bucket, Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=chunk,
            )
            parts.append({"ETag": resp["ETag"], "PartNumber": part_number})
            part_number += 1

        s3_client.complete_multipart_upload(
            Bucket=bucket, Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception as exc:
        logger.error("Multipart upload failed for %s/%s: %s", bucket, key, exc)
        s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise


def _download_from_s3_or_url(uri: str, dest_path: str):
    """Download video from S3 URI or presigned URL."""
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        s3_client.download_file(bucket, key, dest_path)
    else:
        parsed = urlparse(uri)
        host = parsed.hostname or ""
        if host.endswith(".amazonaws.com"):
            bucket, key = _parse_s3_uri(uri)
            s3_client.download_file(bucket, key, dest_path)
        else:
            # treat as presigned / direct URL — download via urllib
            import urllib.request
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
            urllib.request.urlretrieve(uri, dest_path)


# =========================================================================
# SHARED: LETTERBOX PREPROCESS
# =========================================================================
def preprocess(frame: np.ndarray):
    h, w, _ = frame.shape
    scale = min(TARGET_SIZE / h, TARGET_SIZE / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), PAD_VALUE, dtype=np.uint8)

    px = (TARGET_SIZE - nw) // 2
    py = (TARGET_SIZE - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized

    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]

    return img, {"scale": scale, "pad_x": px, "pad_y": py, "orig_w": w, "orig_h": h}


def apply_class_nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes]
    keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=iou_thres)
    return keep.flatten().tolist() if len(keep) > 0 else []


# =========================================================================
# PPE: TRACKING + INFERENCE  (from Triton_PPE.py)
# =========================================================================
@dataclass
class TrackedPerson:
    track_id: int
    box: Tuple[int, int, int, int]
    ppe_history: Dict[str, Deque[Optional[bool]]] = field(default_factory=dict)
    ppe_confidence: Dict[str, Deque[float]] = field(default_factory=dict)
    frames_missing: int = 0
    smoothed_box: Tuple[int, int, int, int] = None

    def __post_init__(self):
        for item in PPE_ITEMS:
            self.ppe_history[item] = deque(maxlen=PPE_TRACK_HISTORY)
            self.ppe_confidence[item] = deque(maxlen=PPE_TRACK_HISTORY)
        self.smoothed_box = self.box

    def update_box(self, new_box):
        if self.smoothed_box is None:
            self.smoothed_box = new_box
        else:
            self.smoothed_box = tuple(
                int(PPE_BOX_SMOOTH_ALPHA * n + (1 - PPE_BOX_SMOOTH_ALPHA) * o)
                for n, o in zip(new_box, self.smoothed_box)
            )
        self.box = new_box
        self.frames_missing = 0

    def add_ppe_observation(self, item, detected, confidence=1.0):
        self.ppe_history[item].append(detected)
        self.ppe_confidence[item].append(confidence)

    def fill_missing_ppe_observations(self, observed_items):
        for item in PPE_ITEMS:
            if item not in observed_items:
                self.ppe_history[item].append(None)
                self.ppe_confidence[item].append(0.0)

    def get_stable_ppe_status(self):
        status = {}
        for item in PPE_ITEMS:
            history = list(self.ppe_history[item])
            confidences = list(self.ppe_confidence[item])
            if not history:
                status[item] = False
                continue
            weighted_true = weighted_false = total_weight = 0.0
            for val, conf in zip(history, confidences):
                if val is None:
                    continue
                weight = max(conf, 0.1)
                total_weight += weight
                if val:
                    weighted_true += weight
                else:
                    weighted_false += weight
            status[item] = (weighted_true / total_weight >= PPE_VOTE_THRESHOLD) if total_weight else False
        return status

    def mark_missing(self):
        self.frames_missing += 1

    def is_expired(self):
        return self.frames_missing > PPE_MISSING_TOLERANCE


class PersonTracker:
    def __init__(self):
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.next_id = 0
        self.per_frame_person_counts: List[int] = []

    def _iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    def _is_inside(self, inner, outer):
        xA, yA = max(inner[0], outer[0]), max(inner[1], outer[1])
        xB, yB = min(inner[2], outer[2]), min(inner[3], outer[3])
        if xB <= xA or yB <= yA:
            return False
        inter_area = (xB - xA) * (yB - yA)
        inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
        return inter_area / inner_area > 0.5 if inner_area > 0 else False

    def _get_vertical_position(self, ppe_box, person_box):
        ppe_center_y = (ppe_box[1] + ppe_box[3]) / 2
        person_height = person_box[3] - person_box[1]
        person_top = person_box[1]
        relative_y = (ppe_center_y - person_top) / person_height if person_height > 0 else 0.5
        if relative_y < 0.33:
            return "top"
        elif relative_y < 0.66:
            return "middle"
        return "bottom"

    def update(self, detected_persons, ppe_detections):
        for p in self.tracked_persons.values():
            p.mark_missing()

        matched_track_ids, matched_det_indices = set(), set()
        matches = []
        for di, db in enumerate(detected_persons):
            for tid, t in self.tracked_persons.items():
                s = self._iou(db, t.smoothed_box)
                if s > PPE_IOU_MATCH_THRESHOLD:
                    matches.append((s, di, tid))
        matches.sort(reverse=True, key=lambda x: x[0])

        for s, di, tid in matches:
            if di in matched_det_indices or tid in matched_track_ids:
                continue
            self.tracked_persons[tid].update_box(detected_persons[di])
            matched_track_ids.add(tid)
            matched_det_indices.add(di)

        for di, db in enumerate(detected_persons):
            if di not in matched_det_indices:
                np_ = TrackedPerson(track_id=self.next_id, box=db)
                self.tracked_persons[self.next_id] = np_
                self.next_id += 1

        self._associate_ppe(ppe_detections)
        self.per_frame_person_counts.append(len(detected_persons))
        expired = [tid for tid, p in self.tracked_persons.items() if p.is_expired()]
        for tid in expired:
            del self.tracked_persons[tid]
        return [p for p in self.tracked_persons.values() if p.frames_missing == 0]

    def get_mode_person_count(self):
        if not self.per_frame_person_counts:
            return 0
        return Counter(self.per_frame_person_counts).most_common(1)[0][0]

    def _associate_ppe(self, ppe_detections):
        person_observed_ppe = {tid: set() for tid in self.tracked_persons}
        for ppe in ppe_detections:
            ppe_box, ppe_name, ppe_conf = ppe["box"], ppe["name"], ppe.get("confidence", 1.0)
            if ppe_name.startswith("no-"):
                base_item, is_present = ppe_name.replace("no-", ""), False
            else:
                base_item, is_present = ppe_name, True
            if base_item not in PPE_ITEMS:
                continue
            best_pid, best_score = None, 0
            for tid, t in self.tracked_persons.items():
                if t.frames_missing > 0:
                    continue
                pb = t.smoothed_box
                iou_s = self._iou(ppe_box, pb)
                inside = self._is_inside(ppe_box, pb)
                score = iou_s + (0.5 if inside else 0)
                if inside:
                    pos = self._get_vertical_position(ppe_box, pb)
                    if base_item in ["helmet", "glasses"] and pos == "top":
                        score += 0.3
                    elif base_item == "vest" and pos == "middle":
                        score += 0.3
                    elif base_item == "shoes" and pos == "bottom":
                        score += 0.3
                if score > best_score:
                    best_score, best_pid = score, tid
            if best_pid is not None and best_score > 0.1:
                self.tracked_persons[best_pid].add_ppe_observation(base_item, is_present, ppe_conf)
                person_observed_ppe[best_pid].add(base_item)
        for tid, t in self.tracked_persons.items():
            if t.frames_missing == 0:
                t.fill_missing_ppe_observations(person_observed_ppe[tid])


def draw_ppe_annotations(frame, tracked_persons, fw, fh):
    for person in tracked_persons:
        x1, y1, x2, y2 = person.smoothed_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        ppe_status = person.get_stable_ppe_status()
        full_ppe = all(ppe_status.values())
        box_color = (0, 255, 0) if full_ppe else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        font, font_scale, thickness, padding, line_h = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, 6, 22
        lines = [(f"{DISPLAY_NAME[it]}: {'Y' if ppe_status[it] else 'N'}", ppe_status[it]) for it in PPE_ITEMS]
        max_w = max(cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t, _ in lines)
        block_h = len(lines) * line_h + padding
        block_w = max_w + padding * 2
        bx1 = max(0, x1)
        by1 = max(0, y1 - block_h - 5)
        bx2 = min(fw, bx1 + block_w)
        by2 = by1 + block_h
        if bx2 >= fw:
            bx1 = max(0, fw - block_w)
            bx2 = fw
        if by2 > by1 and bx2 > bx1:
            overlay = frame[by1:by2, bx1:bx2].copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame[by1:by2, bx1:bx2], 0.3, 0, frame[by1:by2, bx1:bx2])
        y = by1 + line_h
        for text, ok in lines:
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(frame, text, (bx1 + padding, y), font, font_scale, color, thickness)
            y += line_h
    return frame


# =========================================================================
# TRAFFIC: TRACKING + INFERENCE  (from Triton_Traffic.py)
# =========================================================================
@dataclass
class TrackedObject:
    track_id: int
    object_class: str
    box: Tuple[int, int, int, int]
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=TRAFFIC_TRACK_HISTORY))
    frames_missing: int = 0
    smoothed_box: Tuple[int, int, int, int] = None
    first_seen_frame: int = 0
    last_seen_frame: int = 0

    def __post_init__(self):
        self.smoothed_box = self.box

    def update_box(self, new_box, confidence, frame_num):
        if self.smoothed_box is None:
            self.smoothed_box = new_box
        else:
            self.smoothed_box = tuple(
                int(TRAFFIC_BOX_SMOOTH_ALPHA * n + (1 - TRAFFIC_BOX_SMOOTH_ALPHA) * o)
                for n, o in zip(new_box, self.smoothed_box)
            )
        self.box = new_box
        self.confidence_history.append(confidence)
        self.frames_missing = 0
        self.last_seen_frame = frame_num

    def get_average_confidence(self):
        return sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0.0

    def mark_missing(self):
        self.frames_missing += 1

    def is_expired(self):
        return self.frames_missing > TRAFFIC_MISSING_TOLERANCE


class TrafficObjectTracker:
    def __init__(self):
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.frame_count = 0
        self.class_statistics: Dict[str, int] = {cls: 0 for cls in TRAFFIC_CLASSES.values()}
        self.per_frame_class_counts: Dict[str, List[int]] = {cls: [] for cls in TRAFFIC_CLASSES.values()}

    def _iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        self.frame_count += 1
        for obj in self.tracked_objects.values():
            obj.mark_missing()
        matched_track_ids, matched_det_indices = set(), set()
        matches = []
        for di, det in enumerate(detections):
            db, dc = det["box"], det["class"]
            for tid, t in self.tracked_objects.items():
                if t.object_class != dc:
                    continue
                s = self._iou(db, t.smoothed_box)
                if s > TRAFFIC_IOU_MATCH_THRESHOLD:
                    matches.append((s, di, tid))
        matches.sort(reverse=True, key=lambda x: x[0])
        for s, di, tid in matches:
            if di in matched_det_indices or tid in matched_track_ids:
                continue
            d = detections[di]
            self.tracked_objects[tid].update_box(d["box"], d["confidence"], self.frame_count)
            matched_track_ids.add(tid)
            matched_det_indices.add(di)
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                cn = det["class"]
                no = TrackedObject(track_id=self.next_id, object_class=cn,
                                   box=det["box"], first_seen_frame=self.frame_count)
                no.update_box(det["box"], det["confidence"], self.frame_count)
                self.tracked_objects[self.next_id] = no
                self.class_statistics[cn] += 1
                self.next_id += 1
        expired = [tid for tid, o in self.tracked_objects.items() if o.is_expired()]
        for tid in expired:
            del self.tracked_objects[tid]
        active = [o for o in self.tracked_objects.values() if o.frames_missing == 0]
        frame_counts = {}
        for o in active:
            frame_counts[o.object_class] = frame_counts.get(o.object_class, 0) + 1
        for cls in self.per_frame_class_counts:
            self.per_frame_class_counts[cls].append(frame_counts.get(cls, 0))
        return active

    def get_mode_class_counts(self):
        mode_counts = {}
        for cls, counts in self.per_frame_class_counts.items():
            if counts:
                mode_counts[cls] = Counter(counts).most_common(1)[0][0]
            else:
                mode_counts[cls] = 0
        return mode_counts

    def get_statistics(self):
        return {
            "total_tracked": self.next_id,
            "currently_active": len([o for o in self.tracked_objects.values() if o.frames_missing == 0]),
            "class_counts": self.class_statistics.copy(),
            "mode_class_counts": self.get_mode_class_counts(),
        }


def draw_traffic_annotations(frame, tracked_objects, fw, fh, show_track_id=True):
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.smoothed_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        color = TRAFFIC_COLORS.get(obj.object_class, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        avg_conf = obj.get_average_confidence()
        label = f"ID:{obj.track_id} {obj.object_class} {avg_conf:.2f}" if show_track_id else f"{obj.object_class} {avg_conf:.2f}"
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        ly1 = max(0, y1 - lh - baseline - 5)
        ly2 = y1
        lx1 = x1
        lx2 = min(fw, x1 + lw + 10)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    return frame


def draw_traffic_statistics_panel(frame, tracker, fw, fh):
    stats = tracker.get_statistics()
    mode_counts = stats["mode_class_counts"]
    panel_width = 350
    panel_x = fw - panel_width - 10
    panel_y = 10
    line_height = 25
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    lines = [
        f"Frame: {tracker.frame_count}",
        f"Active Objects: {stats['currently_active']}",
        f"Total Tracked: {stats['total_tracked']}",
        "--- Class Counts (Mode) ---"
    ]
    for cls, count in sorted(stats["class_counts"].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            mode_val = mode_counts.get(cls, 0)
            lines.append(f"{cls}: {count} (mode: {mode_val})")
    panel_height = len(lines) * line_height + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
    y = panel_y + line_height
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i < 3 else (255, 255, 255)
        if "---" in line:
            color = (0, 255, 0)
        cv2.putText(frame, line, (panel_x + 10, y), font, font_scale, color, thickness)
        y += line_height
    return frame


# =========================================================================
# TRITON INFERENCE HELPERS
# =========================================================================
def _get_triton_client():
    return httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)


def triton_infer(client, model_name, img):
    inp = InferInput("images", img.shape, "FP32")
    inp.set_data_from_numpy(img)
    out = InferRequestedOutput("output0")
    resp = client.infer(model_name, inputs=[inp], outputs=[out])
    return resp.as_numpy("output0")


def postprocess_ppe(output, meta):
    dets = []
    for p in output[0]:
        x1, y1, x2, y2, conf, cls = p
        cls = int(cls)
        if conf < min(PERSON_CONF_THRES, PPE_CONF_THRES):
            continue
        x1 = max(0, min(meta["orig_w"], (x1 - meta["pad_x"]) / meta["scale"]))
        x2 = max(0, min(meta["orig_w"], (x2 - meta["pad_x"]) / meta["scale"]))
        y1 = max(0, min(meta["orig_h"], (y1 - meta["pad_y"]) / meta["scale"]))
        y2 = max(0, min(meta["orig_h"], (y2 - meta["pad_y"]) / meta["scale"]))
        dets.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "conf": float(conf), "cls": cls})
    return dets


def postprocess_traffic(output, meta):
    dets = []
    for p in output[0]:
        x1, y1, x2, y2, conf, cls = p
        cls = int(cls)
        if conf < TRAFFIC_CONF_THRES or cls not in TRAFFIC_CLASSES:
            continue
        x1 = max(0, min(meta["orig_w"], (x1 - meta["pad_x"]) / meta["scale"]))
        x2 = max(0, min(meta["orig_w"], (x2 - meta["pad_x"]) / meta["scale"]))
        y1 = max(0, min(meta["orig_h"], (y1 - meta["pad_y"]) / meta["scale"]))
        y2 = max(0, min(meta["orig_h"], (y2 - meta["pad_y"]) / meta["scale"]))
        dets.append({"box": (int(x1), int(y1), int(x2), int(y2)),
                      "class": TRAFFIC_CLASSES[cls], "confidence": float(conf)})
    return dets


# =========================================================================
# VIDEO CONVERSION (web-compatible H.264)
# =========================================================================
def convert_video_to_web_format(input_path: str, output_path: str) -> bool:
    try:
        if not shutil.which('ffmpeg'):
            logger.warning("⚠️ FFmpeg not found, skipping conversion")
            return False

        command = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]

        logger.info("🔄 Converting video to web-compatible format...")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_CONVERSION_TIMEOUT,
            check=False
        )

        if result.returncode == 0 and os.path.exists(output_path):
            logger.info("✅ Video conversion successful")
            return True
        else:
            logger.error(f"❌ FFmpeg conversion failed: {result.stderr.decode()[:200]}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ FFmpeg conversion timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Video conversion error: {str(e)}")
        return False


# =========================================================================
# VIDEO PROCESSING PIPELINES
# =========================================================================
def _process_ppe_video(video_path: str, output_path: str, frame_skip: int):
    """Process video with PPE model. Returns (total_frames, duration_seconds, per_person_summary)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"📹 Video info: {W}x{H} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s duration")
    logger.info(f"⚙️ Frame skip: {frame_skip} (processing every {frame_skip + 1} frame(s))")

    # Keep original FPS, but write only processed frames → output is shorter
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    client = _get_triton_client()
    tracker = PersonTracker()

    frames_processed = 0
    frames_skipped = 0
    extraction_time_total = 0.0
    inference_time_total = 0.0
    annotation_time_total = 0.0

    for idx in range(total_frames):
        frame_extract_start = time.time()
        ret, frame = cap.read()
        extraction_time_total += time.time() - frame_extract_start
        if not ret:
            break

        if frame_skip > 0 and idx % (frame_skip + 1) != 0:
            frames_skipped += 1
            continue

        infer_start = time.time()
        img, meta = preprocess(frame)
        output = triton_infer(client, "person_ppe_astec", img)
        dets = postprocess_ppe(output, meta)
        inference_time_total += time.time() - infer_start

        raw_person_boxes, raw_person_scores, ppe_by_class = [], [], {}
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            confidence, cls_id = det["conf"], det["cls"]
            name = PPE_CLASS_MAP.get(cls_id, str(cls_id))
            if name == "person" and confidence >= PERSON_CONF_THRES:
                raw_person_boxes.append((x1, y1, x2, y2))
                raw_person_scores.append(confidence)
            elif (name in PPE_PRESENT_CLASSES or name in PPE_ABSENT_CLASSES) and confidence >= PPE_CONF_THRES:
                ppe_by_class.setdefault(name, {"boxes": [], "scores": []})
                ppe_by_class[name]["boxes"].append((x1, y1, x2, y2))
                ppe_by_class[name]["scores"].append(confidence)

        keep = apply_class_nms(raw_person_boxes, raw_person_scores, PERSON_NMS_IOU)
        detected_persons = [raw_person_boxes[i] for i in keep]

        ppe_detections = []
        for name, data in ppe_by_class.items():
            kp = apply_class_nms(data["boxes"], data["scores"], PPE_NMS_IOU)
            for i in kp:
                ppe_detections.append({"box": data["boxes"][i], "name": name, "confidence": data["scores"][i]})

        annot_start = time.time()
        tracked = tracker.update(detected_persons, ppe_detections)
        frame = draw_ppe_annotations(frame, tracked, W, H)
        annotation_time_total += time.time() - annot_start
        out.write(frame)
        frames_processed += 1

        if frames_processed % PROGRESS_LOG_INTERVAL == 0:
            logger.info(f"🔍 PPE processing: {frames_processed}/{total_frames} frames done "
                        f"({len(detected_persons)} persons, {len(ppe_detections)} PPE items in current frame)")

    cap.release()
    out.release()

    logger.info(f"📊 PPE Frame extraction time:  {extraction_time_total:.2f}s")
    logger.info(f"📊 PPE Detection/inference time: {inference_time_total:.2f}s")
    logger.info(f"📊 PPE Annotation time:          {annotation_time_total:.2f}s")
    logger.info(f"📊 PPE Frames processed: {frames_processed}, skipped: {frames_skipped}")

    # Build per-person PPE summary from all tracked persons (including expired)
    all_persons = {}
    unique_persons = tracker.next_id

    per_person_ppe = {}
    ppe_summary = {"hard_hat": 0, "goggles": 0, "safety_vest": 0,
                   "gloves": 0, "safety_shoes": 0, "persons_detected": unique_persons}

    for tid, p in tracker.tracked_persons.items():
        status = p.get_stable_ppe_status()
        person_label = str(tid + 1)
        per_person_ppe[person_label] = {
            "hardhat": status.get("helmet", False),
            "goggles": status.get("glasses", False),
            "gloves": status.get("gloves", False),
            "shoes": status.get("shoes", False),
            "safety_vest": status.get("vest", False),
            "PPE": all(status.values()),
        }
        if status.get("helmet"):
            ppe_summary["hard_hat"] += 1
        if status.get("glasses"):
            ppe_summary["goggles"] += 1
        if status.get("vest"):
            ppe_summary["safety_vest"] += 1
        if status.get("gloves"):
            ppe_summary["gloves"] += 1
        if status.get("shoes"):
            ppe_summary["safety_shoes"] += 1

    mode_persons = tracker.get_mode_person_count()

    return {
        "total_frames": total_frames,
        "duration_seconds": round(duration, 1),
        "unique_counts": {"persons": unique_persons},
        "mode_persons_per_frame": mode_persons,
        "per_person_ppe_summary": per_person_ppe,
        "ppe_summary": ppe_summary,
    }


def _process_traffic_video(video_path: str, output_path: str, frame_skip: int):
    """Process video with traffic/person detection model."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"📹 Video info: {W}x{H} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s duration")
    logger.info(f"⚙️ Frame skip: {frame_skip} (processing every {frame_skip + 1} frame(s))")

    # Keep original FPS, but write only processed frames → output is shorter
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    client = _get_triton_client()
    tracker = TrafficObjectTracker()

    frames_processed = 0
    frames_skipped = 0
    extraction_time_total = 0.0
    inference_time_total = 0.0
    annotation_time_total = 0.0

    for idx in range(total_frames):
        frame_extract_start = time.time()
        ret, frame = cap.read()
        extraction_time_total += time.time() - frame_extract_start
        if not ret:
            break

        if frame_skip > 0 and idx % (frame_skip + 1) != 0:
            frames_skipped += 1
            continue

        infer_start = time.time()
        img, meta = preprocess(frame)
        output = triton_infer(client, "person_detection_yolo26", img)
        detections = postprocess_traffic(output, meta)
        inference_time_total += time.time() - infer_start

        annot_start = time.time()
        tracked = tracker.update(detections)
        frame = draw_traffic_annotations(frame, tracked, W, H, show_track_id=True)
        frame = draw_traffic_statistics_panel(frame, tracker, W, H)
        annotation_time_total += time.time() - annot_start
        out.write(frame)
        frames_processed += 1

        if frames_processed % PROGRESS_LOG_INTERVAL == 0:
            logger.info(f"🔍 Traffic processing: {frames_processed}/{total_frames} frames done "
                        f"({len(detections)} detections in current frame)")

    cap.release()
    out.release()

    logger.info(f"📊 Traffic Frame extraction time:  {extraction_time_total:.2f}s")
    logger.info(f"📊 Traffic Detection/inference time: {inference_time_total:.2f}s")
    logger.info(f"📊 Traffic Annotation time:          {annotation_time_total:.2f}s")
    logger.info(f"📊 Traffic Frames processed: {frames_processed}, skipped: {frames_skipped}")

    stats = tracker.get_statistics()
    return {
        "total_frames": total_frames,
        "duration_seconds": round(duration, 1),
        "class_counts": stats["class_counts"],
        "mode_class_counts": stats["mode_class_counts"],
    }


# =========================================================================
# API ENDPOINTS
# =========================================================================
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload an MP4 video to S3 via multipart upload. Returns S3 URI and presigned URL."""
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted.")

    start = time.time()
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', os.path.basename(file.filename))
    s3_key = f"uploads/{safe_filename}"

    try:
        file.file.seek(0)
        await run_blocking(_multipart_upload, S3_BUCKET, s3_key, file.file, "video/mp4")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    elapsed = round(time.time() - start, 2)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    presigned_url = _generate_presigned_url(S3_BUCKET, s3_key)

    return {
        "s3_uri": s3_uri,
        "presigned_url": presigned_url,
        "upload_time_seconds": elapsed,
    }


def _process_video_sync(req: ProcessVideoRequest):
    """Synchronous heavy processing function (runs in thread pool)."""
    overall_start = time.time()
    logger.info("=" * 60)
    logger.info("📥 New /process_video request received")
    logger.info(f"   Model:      {req.model_id}")
    logger.info(f"   Video URI:  {req.video_uri}")
    logger.info(f"   Frame skip: {req.frame_skip}")
    logger.info("=" * 60)

    if req.model_id not in ("person_detection_yolo26", "person_ppe_astec"):
        raise HTTPException(status_code=400, detail="model_id must be 'person_detection_yolo26' or 'person_ppe_astec'.")
    if req.frame_skip < 0:
        raise HTTPException(status_code=400, detail="frame_skip must be >= 0.")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "annotated_output.mp4")
        web_video = os.path.join(tmpdir, "annotated_web.mp4")

        # ── Download ──
        logger.info("⬇️ Downloading video from URI...")
        download_start = time.time()
        _download_from_s3_or_url(req.video_uri, local_video)
        download_time = round(time.time() - download_start, 2)
        file_size_mb = round(os.path.getsize(local_video) / (1024 * 1024), 2)
        logger.info(f"✅ Download complete: {file_size_mb} MB in {download_time}s")

        # Determine original filename for output key
        try:
            _, orig_key = _parse_s3_uri(req.video_uri)
            base_name = os.path.splitext(os.path.basename(orig_key))[0]
        except Exception:
            base_name = "video"

        # ── Process (detection + annotation) ──
        logger.info(f"🚀 Starting {req.model_id} processing...")
        processing_start = time.time()
        if req.model_id == "person_ppe_astec":
            result = _process_ppe_video(local_video, output_video, req.frame_skip)
        else:
            result = _process_traffic_video(local_video, output_video, req.frame_skip)
        processing_time = round(time.time() - processing_start, 2)
        logger.info(f"✅ Detection & annotation complete in {processing_time}s")

        # ── Convert to web-compatible format ──
        conversion_start = time.time()
        upload_path = output_video
        converted = convert_video_to_web_format(output_video, web_video)
        conversion_time = round(time.time() - conversion_start, 2)
        if converted:
            upload_path = web_video
            logger.info(f"✅ Web conversion complete in {conversion_time}s")
        else:
            logger.warning(f"⚠️ Web conversion skipped/failed ({conversion_time}s), uploading original annotated video")

        # ── Upload annotated video to S3 ──
        output_key = f"outputs/{base_name}_annotated.mp4"
        logger.info(f"⬆️ Uploading annotated video to s3://{S3_OUTPUT_BUCKET}/{output_key} ...")
        upload_start = time.time()
        with open(upload_path, "rb") as f:
            _multipart_upload(S3_OUTPUT_BUCKET, output_key, f, content_type="video/mp4")
        upload_time = round(time.time() - upload_start, 2)
        logger.info(f"✅ Upload complete in {upload_time}s")

        output_s3_uri = f"s3://{S3_OUTPUT_BUCKET}/{output_key}"
        output_presigned = _generate_presigned_url(S3_OUTPUT_BUCKET, output_key)

        # Determine input type
        input_type = "s3_uri" if req.video_uri.startswith("s3://") else "presigned_url"

        overall_time = round(time.time() - overall_start, 2)

        response = {
            "video_info": {
                "input_uri": req.video_uri,
                "input_type": input_type,
                "output_s3_uri": output_s3_uri,
                "output_presigned_url": output_presigned,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "total_frames": result["total_frames"],
                "duration_seconds": result["duration_seconds"],
                "processing_time_seconds": processing_time,
            },
            "timing_metrics": {
                "download_seconds": download_time,
                "processing_seconds": processing_time,
                "web_conversion_seconds": conversion_time,
                "web_conversion_success": converted,
                "upload_seconds": upload_time,
                "overall_seconds": overall_time,
            },
        }

        if req.model_id == "person_ppe_astec":
            response["unique_counts"] = result["unique_counts"]
            response["mode_persons_per_frame"] = result["mode_persons_per_frame"]
            response["per_person_ppe_summary"] = result["per_person_ppe_summary"]
            response["ppe_summary"] = result["ppe_summary"]
        else:
            response["class_counts"] = result["class_counts"]
            response["mode_class_counts"] = result["mode_class_counts"]

    return response


@app.post("/process_video")
async def process_video(req: ProcessVideoRequest):
    """Async wrapper that safely runs heavy processing concurrently."""
    try:
        return await run_blocking(_process_video_sync, req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Test health of both Triton model endpoints."""
    client = _get_triton_client()
    results = {}

    for model_name in ("person_detection_yolo26", "person_ppe_astec"):
        try:
            alive = client.is_model_ready(model_name)
            results[model_name] = {"status": "healthy" if alive else "not_ready", "ready": alive}
        except Exception as e:
            results[model_name] = {"status": "unreachable", "ready": False, "error": str(e)}

    try:
        server_live = client.is_server_live()
        server_ready = client.is_server_ready()
    except Exception:
        server_live = False
        server_ready = False

    overall = server_live and server_ready and all(r["ready"] for r in results.values())

    return {
        "triton_server": {"live": server_live, "ready": server_ready},
        "models": results,
        "overall_healthy": overall,
    }


# =========================================================================
# ENTRYPOINT
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=8092, workers=10)
