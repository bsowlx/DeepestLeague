import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict


def detect_viewport_robust(img: np.ndarray, visualize: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """Detect the observer viewport rectangle on a minimap crop.

    Returns (x, y, w, h) in pixel coordinates, or None if no viewport found.
    Uses a three-strategy cascade: corner junctions → partial-edge inference → longest-line fallback.
    """
    if img is None:
        return None
    
    h_img, w_img = img.shape[:2]
    
    ASPECT_RATIO = 16.0 / 9.0
    MIN_W = w_img * 0.15
    MIN_H = MIN_W / ASPECT_RATIO
    BORDER_MARGIN = 5
    
    # --- Preprocessing ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    brightness_threshold = np.percentile(v_channel, 98)
    thresh_val = max(160, min(230, brightness_threshold - 10))
    
    mask = cv2.inRange(hsv, np.array([0, 0, int(thresh_val)]), np.array([180, 60, 255]))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    edges = cv2.Canny(mask, 50, 150)
    
    if visualize:
        cv2.imshow('Mask', mask)
        cv2.imshow('Edges', edges)
    
    # --- Hough line detection (progressive relaxation) ---
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                           minLineLength=15, maxLineGap=10)
    
    if lines is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10,
                               minLineLength=10, maxLineGap=20)
    
    if lines is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=5,
                               minLineLength=8, maxLineGap=30)
    
    if lines is None:
        return None
    
    # --- Geometric parsing ---
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 8:
            continue
        
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 15 or angle > 165:
            h_lines.append({
                'y': (y1 + y2) / 2,
                'x1': min(x1, x2),
                'x2': max(x1, x2),
                'len': length,
                'orig': (x1, y1, x2, y2)
            })
        elif 75 < angle < 105:
            v_lines.append({
                'x': (x1 + x2) / 2,
                'y1': min(y1, y2),
                'y2': max(y1, y2),
                'len': length,
                'orig': (x1, y1, x2, y2)
            })
    
    if visualize:
        vis_lines = img.copy()
        for h in h_lines:
            cv2.line(vis_lines, (int(h['x1']), int(h['y'])), 
                    (int(h['x2']), int(h['y'])), (0, 255, 0), 2)
        for v in v_lines:
            cv2.line(vis_lines, (int(v['x']), int(v['y1'])), 
                    (int(v['x']), int(v['y2'])), (255, 0, 0), 2)
        cv2.imshow('Lines', vis_lines)
    
    if not h_lines and not v_lines:
        return None
    
    # --- Strategy 1: Corner junction ---
    best_score = -1
    best_box = None
    
    for h in h_lines:
        for v in v_lines:
            dist_thresh = 12
            
            junctions = [
                (h['x1'], h['y'], 'TL') if abs(h['x1'] - v['x']) < dist_thresh and abs(h['y'] - v['y1']) < dist_thresh else None,
                (h['x2'], h['y'], 'TR') if abs(h['x2'] - v['x']) < dist_thresh and abs(h['y'] - v['y1']) < dist_thresh else None,
                (h['x1'], h['y'], 'BL') if abs(h['x1'] - v['x']) < dist_thresh and abs(h['y'] - v['y2']) < dist_thresh else None,
                (h['x2'], h['y'], 'BR') if abs(h['x2'] - v['x']) < dist_thresh and abs(h['y'] - v['y2']) < dist_thresh else None,
            ]
            
            valid_junctions = [j for j in junctions if j is not None]
            
            for (cx, cy, c_type) in valid_junctions:
                # Detect clipping
                h_other_x = h['x2'] if 'L' in c_type else h['x1']
                h_clipped = (h_other_x < BORDER_MARGIN) or (h_other_x > w_img - BORDER_MARGIN)
                
                v_other_y = v['y2'] if 'T' in c_type else v['y1']
                v_clipped = (v_other_y < BORDER_MARGIN) or (v_other_y > h_img - BORDER_MARGIN)
                
                # Calculate dimensions
                if not h_clipped and not v_clipped:
                    est_w = h['len']
                    est_h = v['len']
                elif h_clipped and not v_clipped:
                    est_h = v['len']
                    est_w = est_h * ASPECT_RATIO
                elif not h_clipped and v_clipped:
                    est_w = h['len']
                    est_h = est_w / ASPECT_RATIO
                else:
                    est_w = max(h['len'], v['len'] * ASPECT_RATIO)
                    est_h = est_w / ASPECT_RATIO
                
                # Project box
                dx = 1 if 'L' in c_type else -1
                dy = 1 if 'T' in c_type else -1
                
                final_x = cx if dx == 1 else cx - est_w
                final_y = cy if dy == 1 else cy - est_h
                
                # Scoring
                score = h['len'] + v['len']
                
                if not h_clipped and not v_clipped:
                    ratio = est_w / (est_h + 1e-5)
                    if 1.6 < ratio < 1.9:
                        score += 30
                else:
                    score += 10
                
                if est_w < MIN_W:
                    score -= 100
                
                if score > best_score:
                    best_score = score
                    best_box = (int(round(final_x)), int(round(final_y)),
                               int(round(est_w)), int(round(est_h)))
    
    # --- Strategy 2: Partial-edge inference ---
    if best_box is None and (h_lines or v_lines):
        partial_score = -1
        
        for h in h_lines:
            near_left = h['x1'] < BORDER_MARGIN * 2
            near_right = h['x2'] > w_img - BORDER_MARGIN * 2
            near_top = h['y'] < BORDER_MARGIN * 2
            near_bottom = h['y'] > h_img - BORDER_MARGIN * 2
            
            if not (near_left or near_right or near_top or near_bottom):
                continue
            
            est_w = h['len']
            est_h = est_w / ASPECT_RATIO
            
            if near_top:
                final_y = h['y']
                final_x = h['x1']
            elif near_bottom:
                final_y = h['y'] - est_h
                final_x = h['x1']
            else:
                final_x = h['x1'] if near_left else h['x2'] - est_w
                final_y = max(0, h['y'] - est_h / 2)
            
            if est_w >= MIN_W and est_h >= MIN_H and h['len'] > partial_score:
                partial_score = h['len']
                best_box = (int(final_x), int(final_y), int(est_w), int(est_h))
        
        if best_box is None:
            for v in v_lines:
                near_left = v['x'] < BORDER_MARGIN * 2
                near_right = v['x'] > w_img - BORDER_MARGIN * 2
                near_top = v['y1'] < BORDER_MARGIN * 2
                near_bottom = v['y2'] > h_img - BORDER_MARGIN * 2
                
                if not (near_left or near_right or near_top or near_bottom):
                    continue
                
                est_h = v['len']
                est_w = est_h * ASPECT_RATIO
                
                if near_left:
                    final_x = v['x']
                    final_y = v['y1']
                elif near_right:
                    final_x = v['x'] - est_w
                    final_y = v['y1']
                else:
                    final_y = v['y1'] if near_top else v['y2'] - est_h
                    final_x = max(0, v['x'] - est_w / 2)
                
                if est_w >= MIN_W and est_h >= MIN_H and v['len'] > partial_score:
                    partial_score = v['len']
                    best_box = (int(final_x), int(final_y), int(est_w), int(est_h))
    
    # --- Strategy 3: Longest-line fallback ---
    if best_box is None:
        all_lines = h_lines + v_lines
        if all_lines:
            all_lines.sort(key=lambda k: k['len'], reverse=True)
            best_line = all_lines[0]
            
            if best_line['len'] > w_img * 0.2:
                if best_line in h_lines:
                    est_w = best_line['len']
                    est_h = est_w / ASPECT_RATIO
                    final_x = best_line['x1']
                    final_y = best_line['y'] if best_line['y'] < h_img / 2 else best_line['y'] - est_h
                else:
                    est_h = best_line['len']
                    est_w = est_h * ASPECT_RATIO
                    final_y = best_line['y1']
                    final_x = best_line['x'] if best_line['x'] < w_img / 2 else best_line['x'] - est_w
                
                best_box = (int(final_x), int(final_y), int(est_w), int(est_h))
    
    return best_box


def main():
    parser = argparse.ArgumentParser(description='Stable LoL Minimap Viewport Detection')
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--visualize", action="store_true", help="Show debug windows")
    parser.add_argument("--process-size", type=int, default=256)
    args = parser.parse_args()
    
    in_path = Path(args.input_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    lost_path = out_path / "failures"
    lost_path.mkdir(exist_ok=True)
    
    print(f"Running STABLE Detection on: {in_path}")
    img_files = sorted([f for f in in_path.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    
    if not img_files:
        print(f"No images found in {in_path}")
        return
    
    success_count = 0
    total_count = len(img_files)
    
    for i, img_path in enumerate(img_files, 1):
        print(f"[{i}/{total_count}] {img_path.name}", end=" ... ")
        
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            print("SKIP (read error)")
            continue
        
        # Resize to standard size
        img = cv2.resize(raw_img, (args.process_size, args.process_size), 
                        interpolation=cv2.INTER_AREA)
        
        box = detect_viewport_robust(img, visualize=args.visualize)
        
        vis = img.copy()
        if box:
            success_count += 1
            x, y, w, h = box
            
            # Draw virtual box (blue)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw visible portion (green overlay)
            x_vis = max(0, x)
            y_vis = max(0, y)
            w_vis = min(args.process_size, x + w) - x_vis
            h_vis = min(args.process_size, y + h) - y_vis
            
            if w_vis > 0 and h_vis > 0:
                overlay = vis.copy()
                cv2.rectangle(overlay, (x_vis, y_vis), 
                            (x_vis + w_vis, y_vis + h_vis), 
                            (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            
            # Add info text
            cv2.putText(vis, f"({x},{y},{w},{h})", (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imwrite(str(out_path / img_path.name), vis)
            print("SUCCESS")
        else:
            cv2.putText(vis, "LOST", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(lost_path / img_path.name), vis)
            print("FAILED")
        
        if args.visualize:
            cv2.imshow('Result', vis)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Total: {total_count}")
    print(f"Success: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Rate: {(success_count/total_count)*100:.2f}%")
    print(f"{'='*60}")
    print(f"Results: {out_path}")
    print(f"Failures: {lost_path}")


if __name__ == "__main__":
    main()