"""
spot_perception.py

Perception and geometry utilities for Spot vision and manipulation.

Author: Tim
Date: June 27, 2025
"""

import numpy as np
import math
import cv2
import torch
from typing import List, Tuple

from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

g_image_click = None
g_image_display = None
class SpotPerception:
    """
    Contains computer vision and geometric helper functions for Spot.
    All methods are static, as they don't depend on class state.
    """

    @staticmethod
    def find_grasp_sam(cv_img, depth_img, left, conf=0.15, min_area_frac=0.03,
                       group_adj=True, gap_frac=0.18, pad_frac=0.1,
                       prefer_largest=True, center_bias=0.4, max_distance_m=3.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_ckpt = "./sam_vit_h_4b8939.pth"  #TODO Needs adjusting
        model_type = "vit_h"
        labels = ["cardboard box", "shipping box", "moving box", 
                  "corrugated box", "brown box"]
        
        h, w = cv_img.shape[:2]
        
        # 1) detect
        det_boxes, det_scores, det_names = detect_with_owlv2(cv_img, labels, device, conf)
        if len(det_boxes) == 0:
            raise RuntimeError("No 'cardboard box' detections. Try lowering --conf or adding synonyms in --labels.")

        # filter tiny far-away rectangles
        areas = (det_boxes[:,2]-det_boxes[:,0]) * (det_boxes[:,3]-det_boxes[:,1])
        keep = np.where(areas >= min_area_frac * (h*w))[0]
        det_boxes, det_scores = det_boxes[keep], det_scores[keep]
        det_names = [det_names[i] for i in keep]

        # 2) group adjacent → clusters
        if group_adj:
            clusters = group_adjacent_boxes(det_boxes, det_scores, det_names, gap_frac, h, w)
        else:
            clusters = [{"indices":[i], "box": det_boxes[i], "score": float(det_scores[i]), "count":1}
                        for i in range(len(det_boxes))]
        
        # 3) refine each cluster with SAM (box + seeded positive points)
        sam_masks, infos = [], []
        for c in clusters:
            member_boxes = det_boxes[c["indices"]]
            mask_u8, sam_score, det_iou, roi = refine_with_sam_on_cluster(
                cv_img, c["box"], member_boxes,
                sam_ckpt, model_type, device,
                pad_frac=pad_frac
            )
            sam_masks.append(mask_u8)
            infos.append({"cluster": c, "sam_score": sam_score, "det_iou": det_iou})

        # 4) choose best cluster by composite score
        best = None
        img_center = np.array([w/2.0, h/2.0], dtype=np.float32)
        for i, (m, info) in enumerate(zip(sam_masks, infos)):
            m = mask_largest_component(m)
            rect_score, box = rectangularity_and_box(m)
            if box is None: continue

            area = float(m.sum()) / float(h*w)
            # center prior
            cx, cy = box.mean(axis=0)
            dist = np.linalg.norm(np.array([cx,cy]) - img_center) / math.sqrt(w*w + h*h)
            center_prior = 1.0 - dist  # [0,1]

            # size prior (helps close-up)
            size_prior = area

            # composite score
            sc_det = max(1e-3, info["cluster"]["score"])
            sc_sam = max(1e-3, info["sam_score"])
            sc_iou = max(1e-3, info["det_iou"])

            score = (rect_score ** 0.5) * (sc_det ** 0.4) * (sc_sam ** 0.6) * (sc_iou ** 0.8)
            if prefer_largest:
                score *= (size_prior ** 0.7)
            if center_bias > 0:
                score *= (center_prior ** center_bias)

            if best is None or score > best[0]:
                best = (score, i, box, m)

        if best is None:
            raise RuntimeError("Could not form a stable rectangle from any cluster.")

        score, idx, box, mask = best
        box_ord = order_box_points(box)
        edges = [ (tuple(box_ord[i]), tuple(box_ord[(i+1)%4])) for i in range(4) ]

        # 5) save / print
        out_path = "img_with_sam_overlay.png"
        vis = draw_result(cv_img, mask, box_ord)
        cv2.imwrite(out_path, vis)

        cinfo = infos[idx]["cluster"]
        print("=== Selected cluster ===")
        print(f"members={cinfo['count']} det_score_mean={cinfo['score']:.3f} sam_iou={infos[idx]['det_iou']:.3f}")
        print(f"Composite score: {score:.3f}")
        print("\nCorners (ordered circularly):")
        for i, (x,y) in enumerate(box_ord):
            print(f"P{i}: ({x:.1f}, {y:.1f})")
        print("\nEdges (point pairs):")
        for i, ((x1,y1),(x2,y2)) in enumerate(edges):
            print(f"E{i}: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
        print(f"\nSaved visualization to: {out_path}")

        #If left robot get left edge mid point, if right robot get right edge
        if left:
            mid_pt = [(box_ord[0][0]+box_ord[3][0])/2, (box_ord[0][1]+box_ord[3][1])/2]
        else:
            mid_pt = [(box_ord[1][0]+box_ord[2][0])/2, (box_ord[1][1]+box_ord[2][1])/2]

        # depth, px, py = SpotPerception.get_depth_at_pixel(depth_img, mid_pt[0], mid_pt[1], search_radius=20)
        # if depth is not None and depth < max_distance_m:
        #     return px, py
        
        return mid_pt

    @staticmethod
    def find_strongest_vertical_edge(cv_img, depth_img):
        """
        Finds the strongest vertical edge in the input image.

        Args:
            cv_image (np.ndarray): Input color or grayscale image.
        Returns:
            edge_x (int): The x-coordinate of the strongest vertical edge.
        
        #TODO Update to help the robots know if they are left or right edge
        #TODO Use depth to help figure out what edges are the box
        """
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img.copy()        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 
            threshold=50,minLineLength=50, maxLineGap=10
        )
        best_line = None
        best_score = -np.inf
        h = cv_img.shape[0]
        if lines is not None:
            max_distance_m = 3.0  # or whatever you want
            good_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                depth, px, py = SpotPerception.get_depth_at_pixel(depth_img, mid_x, mid_y, search_radius=20)
                if depth is not None and depth < max_distance_m:
                    # Optionally, you can use (px, py) as your grasp point instead of the exact midpoint
                    good_lines.append((line, px, py, depth))
            
            for line, px, py, depth in good_lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                length = np.hypot(dx, dy)
                # Check if vertical enough
                if dx < 0.5*dy and length > 30:
                    # Score: longer, and closer to bottom of image (foreground)
                    avg_y = (y1 + y2) / 2
                    score = avg_y + 2 * length  # Heavily favor lines lower in image
                    if score > best_score:
                        best_score = score
                        best_line = (x1, y1, x2, y2, px, py)

        if not best_line:
            print("best_line is empty")
        return best_line

    @staticmethod
    def get_vertical_edge_grasp_point(visual_img, depth_img, id="spot", 
                                      save_img:bool=False):
        """
        Calculates a grasp point on a detected vertical edge.

        Args:
            visual_img (np.ndarray): Visual image (rgb or gray)
            depth_img (np.ndarray): Depth image
        Returns:
            grasp_point (tuple): (x, y) pixel coordinates for grasp.
        """
        # Find the strongest vertical line
        line = SpotPerception.find_strongest_vertical_edge(visual_img, depth_img)
        if not line:
            print("No strong vertical line found.")
            return None
        x1, y1, x2, y2, px, py = line
        
        if save_img:
            SpotPerception.save_markup_img(visual_img, line, id)
        
        return px, py
    
    @staticmethod
    def save_markup_img(img, line, id, depth=None):
        """
        Save image with edge detection markup.
        """
        img_mark = img.copy()
        x1, y1, x2, y2, px, py = line
        # Calculate line midpoint
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        # Draw the detected vertical edge as a green line
        cv2.line(img_mark, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Draw the grasp pixel (red dot). If valid depth found, use (px,py), else use (cx,cy)
        cv2.circle(img_mark, (px, py), 6, (0, 0, 255), -1)
        
        path = f"images/{id}_edge_and_depth_pixel.png"
        cv2.imwrite(path, img_mark)
        print(f"{id}: Saved edge and grasp visualization to {path}")

    @staticmethod
    def depth_to_point_cloud(depth_img, camera_model, region=None):
        """
        Converts a depth image to a point cloud in camera frame.

        Args:
            depth_img (np.ndarray): Depth image (uint16).
            camera_model: Camera intrinsics/model (from Spot SDK).
            region (tuple, optional): (xmin, xmax, ymin, ymax) crop region.

        Returns:
            points (np.ndarray): Nx3 array of XYZ points.
        """
        # Assume depth in millimeters, shape [H,W]
        if region:
            x1, y1, x2, y2 = region
            depth_crop = depth_img[y1:y2, x1:x2]
            xs, ys = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        else:
            h, w = depth_img.shape
            depth_crop = depth_img
            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        depths = depth_crop.flatten().astype(np.float32) / 1000.0  # mm to meters
        xs = xs.flatten()
        ys = ys.flatten()
        # Camera intrinsics
        fx = camera_model.intrinsics.focal_length.x
        fy = camera_model.intrinsics.focal_length.y
        cx = camera_model.intrinsics.principal_point.x
        cy = camera_model.intrinsics.principal_point.y
        X = (xs - cx) * depths / fx
        Y = (ys - cy) * depths / fy
        Z = depths
        points = np.stack([X, Y, Z], axis=1)
        return points

    @staticmethod
    def fit_plane(points):
        """
        Fits a plane to a set of 3D points.

        Args:
            points (np.ndarray): Nx3 array of 3D points.

        Returns:
            plane_normal (np.ndarray): 3-vector for normal.
            plane_point (np.ndarray): A point on the plane.
        """
        valid = ~np.isnan(points).any(axis=1) & (points[:,2] > 0.2) & (points[:,2] < 3.0)  # filter valid range
        pts = points[valid]
        if pts.shape[0] < 10:
            return None, None
        # Fit plane to points: Ax + By + Cz + D = 0
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        U, S, Vt = np.linalg.svd(pts_centered)
        normal = Vt[-1]
        normal /= np.linalg.norm(normal)
        d = -centroid.dot(normal)
        return normal, d

    @staticmethod
    def get_depth_at_pixel(depth_img, x, y, search_radius=5):
        """
        Gets the depth value at a pixel.

        Args:
            depth_img (np.ndarray): Depth image.
            x (int): Pixel x-coordinate.
            y (int): Pixel y-coordinate.
            search_radius (int): Radius in pixels to search for a valid depth.

        Returns:
            depth (float): Depth value at (x, y).
        """
        h, w = depth_img.shape[:2]
        # Try the center pixel first
        if 0 <= x < w and 0 <= y < h:
            d = depth_img[y, x]
            if np.isfinite(d) and d > 0:
                return (float(d) / 1000.0 if depth_img.dtype == np.uint16 else float(d), x, y)
        # Search in a small window for a valid depth
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    px = x + dx
                    py = y + dy
                    if 0 <= px < w and 0 <= py < h:
                        d = depth_img[py, px]
                        if np.isfinite(d) and d > 0:
                            return (float(d) / 1000.0 if depth_img.dtype == np.uint16 else float(d), px, py)
        return (None, None, None)

    @staticmethod
    def pixel_to_camera_frame(x, y, depth, camera_model):
        """
        Converts pixel coordinates and depth to camera frame XYZ.

        Args:
            x (int): Pixel x-coordinate.
            y (int): Pixel y-coordinate.
            depth (float): Depth at (x, y).
            camera_model: Camera intrinsics/model (from Spot SDK).

        Returns:
            xyz (np.ndarray): 3D point in camera frame.
        """
        # camera_model contains focal_length, center_x, center_y
        fx = camera_model.intrinsics.focal_length.x
        fy = camera_model.intrinsics.focal_length.y
        cx = camera_model.intrinsics.principal_point.x
        cy = camera_model.intrinsics.principal_point.y
        # Unproject
        x = (x - cx) * depth / fx
        y = (y - cy) * depth / fy
        z = depth
        return (x, y, z)

    @staticmethod
    def get_target_from_user(img):
        """
        Displays an image from the robot's camera and waits for user to
        click on a target.
        """
        # Show the image to the user and wait for them to click on a pixel
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, SpotPerception.cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_click = None
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                exit(0)

        print(f"g_image_click {g_image_click}")
        return g_image_click

    @staticmethod
    def cv_mouse_callback(event, x, y, flags, param):
        global g_image_click, g_image_display
        clone = g_image_display.copy()
        if event == cv2.EVENT_LBUTTONUP:
            g_image_click = (x, y)
        else:
            # Draw some lines on the image.
            # print('mouse', x, y)
            color = (30, 30, 30)
            thickness = 2
            image_title = 'Click to grasp'
            height = clone.shape[0]
            width = clone.shape[1]
            cv2.line(clone, (0, y), (width, y), color, thickness)
            cv2.line(clone, (x, 0), (x, height), color, thickness)
            cv2.imshow(image_title, clone)

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:,None] + area_b[None,:] - inter
    return inter / np.clip(union, 1e-6, None)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1: break
        iou = iou_matrix(boxes[i][None, :], boxes[idxs[1:]])[0]
        idxs = idxs[1:][iou < iou_thresh]
    return keep

def interval_gap(a1, a2, b1, b2) -> float:
    # gap between 1D intervals; 0 if overlapping
    if a2 < b1: return b1 - a2
    if b2 < a1: return a1 - b2
    return 0.0

def mask_largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    n, labels = cv2.connectedComponents(m)
    if n <= 1: return (m > 0).astype(np.uint8)
    largest = 1 + np.argmax([(labels == i).sum() for i in range(1, n)])
    return (labels == largest).astype(np.uint8)

def rectangularity_and_box(mask: np.ndarray) -> Tuple[float, np.ndarray]:
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0, None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10: return 0.0, None
    eps = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) == 4:
        box = approx.reshape(-1, 2).astype(np.float32)
        rect_area = max(cv2.contourArea(approx), 1.0)
    else:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        w = np.linalg.norm(box[0] - box[1]); h = np.linalg.norm(box[1] - box[2])
        rect_area = max(w*h, 1.0)
    mask_area = float(m.sum())
    return float(mask_area / rect_area), box

def order_box_points(box: np.ndarray) -> np.ndarray:
    c = box.mean(axis=0)
    ang = np.arctan2(box[:,1] - c[1], box[:,0] - c[0])
    return box[np.argsort(ang)]

def draw_result(img: np.ndarray, mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    vis = img.copy()
    vis[mask > 0] = (0.6 * vis[mask > 0] + 0.4 * np.array([0,255,0])).astype(np.uint8)
    cv2.polylines(vis, [box.astype(np.int32)], True, (0,0,255), 2, cv2.LINE_AA)
    for i, (x,y) in enumerate(box.astype(int)):
        cv2.circle(vis, (x,y), 4, (255,0,0), -1, cv2.LINE_AA)
        cv2.putText(vis, f"P{i}", (x+4,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    return vis

# ========== OWLv2 detection ==========
def detect_with_owlv2(image_bgr: np.ndarray, labels: List[str], device: str, conf_thresh: float):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

    inputs = processor(text=labels, images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_rgb.shape[:2]], device=device)  # (h, w)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    boxes = results["boxes"].detach().cpu().numpy()  # (x1,y1,x2,y2)
    scores = results["scores"].detach().cpu().numpy()
    class_ids = results["labels"].detach().cpu().numpy()

    keep = np.where(scores >= conf_thresh)[0]
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    if len(boxes) == 0:
        return np.zeros((0,4),dtype=np.float32), np.zeros((0,),dtype=np.float32), []

    keep = nms(boxes, scores, iou_thresh=0.5)
    return boxes[keep], scores[keep], [labels[i] for i in class_ids[keep]]

# ========== grouping adjacent detections ==========
def group_adjacent_boxes(boxes: np.ndarray, scores: np.ndarray, names: List[str],
                         gap_frac: float, img_h: int, img_w: int):
    if len(boxes) == 0:
        return []

    hts = boxes[:,3] - boxes[:,1]
    wds = boxes[:,2] - boxes[:,0]
    ref = max(1.0, float(np.median(hts)))
    gap_thresh = gap_frac * ref  # pixel gap allowed to merge

    # union-find
    parent = list(range(len(boxes)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            xi1, yi1, xi2, yi2 = boxes[i]
            xj1, yj1, xj2, yj2 = boxes[j]
            # overlp in Y and small horizontal gap OR overlap in X and small vertical gap OR IoU>0.1
            ovy = max(0, min(yi2, yj2) - max(yi1, yj1))
            ovx = max(0, min(xi2, xj2) - max(xi1, xj1))
            gapx = interval_gap(xi1, xi2, xj1, xj2)
            gapy = interval_gap(yi1, yi2, yj1, yj2)
            iou = iou_matrix(boxes[i][None,:], boxes[j][None,:])[0,0]
            cond = ((ovy / max(1.0, min(yi2-yi1, yj2-yj1)) > 0.4 and gapx <= gap_thresh) or
                    (ovx / max(1.0, min(xi2-xi1, xj2-xj1)) > 0.4 and gapy <= gap_thresh) or
                    (iou > 0.10))
            if cond:
                union(i,j)

    # collect clusters
    groups = {}
    for k in range(len(boxes)):
        r = find(k)
        groups.setdefault(r, []).append(k)

    clusters = []
    for gidx, idxs in groups.items():
        b = boxes[idxs]
        x1 = float(b[:,0].min()); y1 = float(b[:,1].min())
        x2 = float(b[:,2].max()); y2 = float(b[:,3].max())
        sc = float(scores[idxs].mean())
        clusters.append({
            "indices": idxs,
            "box": np.array([x1,y1,x2,y2], dtype=np.float32),
            "score": sc,
            "count": len(idxs),
        })
    return clusters

# ========== SAM refinement with detector constraints ==========
def refine_with_sam_on_cluster(image_bgr: np.ndarray,
                               cluster_box_xyxy: np.ndarray,
                               member_boxes_xyxy: np.ndarray,
                               sam_ckpt: str, model_type: str, device: str,
                               pad_frac: float = 0.08,
                               seed_points: int = 12):
    H, W = image_bgr.shape[:2]
    # pad ROI
    x1,y1,x2,y2 = cluster_box_xyxy.astype(np.float32)
    pw = pad_frac * (x2 - x1); ph = pad_frac * (y2 - y1)
    x1p = max(0, int(round(x1 - pw))); y1p = max(0, int(round(y1 - ph)))
    x2p = min(W-1, int(round(x2 + pw))); y2p = min(H-1, int(round(y2 + ph)))
    roi = np.array([x1p,y1p,x2p,y2p], dtype=np.float32)

    sam = sam_model_registry[model_type](checkpoint=sam_ckpt).to(device)
    predictor = SamPredictor(sam)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # build positive seed points inside the union of member boxes (to keep SAM on-target)
    # sample a coarse grid in ROI and keep points that fall inside any member box
    xs = np.linspace(x1p, x2p, int(max(3, math.sqrt(seed_points))*2))
    ys = np.linspace(y1p, y2p, int(max(3, math.sqrt(seed_points))*2))
    pts = []
    for xx in xs:
        for yy in ys:
            for mb in member_boxes_xyxy:
                if (xx >= mb[0]) and (xx <= mb[2]) and (yy >= mb[1]) and (yy <= mb[3]):
                    pts.append([xx, yy]); break
    if len(pts) == 0:
        # fallback: use center of ROI
        pts = [[(x1p+x2p)/2.0, (y1p+y2p)/2.0]]
    # limit number of points
    pts = np.array(pts, dtype=np.float32)
    if len(pts) > seed_points:
        sel = np.linspace(0, len(pts)-1, seed_points).astype(int)
        pts = pts[sel]

    point_labels = np.ones((len(pts),), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=point_labels,
        box=roi[None, :],
        multimask_output=True
    )

    # choose mask with max IoU to union of member boxes (to avoid background drift)
    union_mask = np.zeros((H, W), dtype=np.uint8)
    for mb in member_boxes_xyxy:
        x1m, y1m, x2m, y2m = mb.astype(int)
        union_mask[y1m:y2m, x1m:x2m] = 1

    best = None
    for i, m in enumerate(masks):
        m_u8 = (m > 0.5).astype(np.uint8)
        inter = (m_u8 & union_mask).sum()
        union = (m_u8 | union_mask).sum()
        iou = float(inter) / float(max(1, union))
        best = (iou, i, m_u8) if (best is None or iou > best[0]) else best

    iou_best, idx, mask_u8 = best
    return mask_u8, float(scores[idx]), float(iou_best), roi