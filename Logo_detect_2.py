# 相较于第一版做了以下优化
# ROI 只在左侧 1/3，HSV 转换区域更小 → 提速。
#
# 相邻聚类 代替 DBSCAN，不依赖 sklearn，逻辑也更快。

import fitz
import cv2
import numpy as np
from PIL import Image
import os

# ================== 配置参数 ==================
PDF_PATH = r"D:\Yy\Documents\yuqing\6.pdf"
OUTPUT_DIR = r"D:\Yy\Documents\yuqing"

DPI = 300
TOP_OFFSET_MM = 10
MIN_LOGO_AREA = 150
CLUSTER_EPS = 15  # 聚类的容忍距离(px)

LOGO_ROI_LEFT_RATIO = 0.0
LOGO_ROI_RIGHT_RATIO = 1.0 / 3.0   # ← 左侧三分之一
OUTPUT_FORMAT = "PNG"
# =============================================

def mm_to_px(mm, dpi):
    return int(dpi * mm / 25.4)

def get_color_masks(hsv):
    """提取蓝色和红色区域的掩码"""
    # 蓝色
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 红色（跨0度）
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    return mask_blue + mask_red

def find_logo_tops(img, dpi):
    """返回每个Logo最上方Y坐标"""
    h, w = img.shape[:2]
    x1 = int(w * LOGO_ROI_LEFT_RATIO)
    x2 = int(w * LOGO_ROI_RIGHT_RATIO)

    roi_hsv = cv2.cvtColor(img[:, x1:x2], cv2.COLOR_RGB2HSV)
    mask = get_color_masks(roi_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_tops = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_LOGO_AREA:
            continue
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        raw_tops.append(y)  # ROI 内的 y 坐标

    if not raw_tops:
        return []

    # ==== 相邻聚类替代 DBSCAN ====
    raw_tops.sort()
    unique_tops = [raw_tops[0]]
    for t in raw_tops[1:]:
        if abs(t - unique_tops[-1]) > CLUSTER_EPS:
            unique_tops.append(t)
    # ===========================
    return unique_tops

def detect_bottom_edge(img_gray, start_y, min_text_density=5):
    h_proj = np.sum(img_gray < 200, axis=1)
    valid_rows = np.where(h_proj > min_text_density)[0]
    if len(valid_rows) == 0:
        return img_gray.shape[0]
    last_text_row = valid_rows[-1]
    return min(img_gray.shape[0], last_text_row + mm_to_px(10, DPI))

def save_cropped_as_image(cropped_img, output_path, format="PNG"):
    pil_img = Image.fromarray(cropped_img)
    if format.upper() == "JPEG":
        pil_img = pil_img.convert("RGB")
    pil_img.save(output_path, format=format, quality=95 if format == "JPEG" else None)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    doc = fitz.open(PDF_PATH)
    receipt_count = 0
    px_offset = mm_to_px(TOP_OFFSET_MM, DPI)

    for page_idx in range(len(doc)):
        print(f"正在处理第 {page_idx + 1} / {len(doc)} 页...")

        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = doc[page_idx].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        logo_tops = find_logo_tops(img_np, DPI)
        if not logo_tops:
            print("  ⚠️ 未检测到Logo，跳过")
            continue

        top_lines = [max(0, top - px_offset) for top in logo_tops]
        top_lines = sorted(top_lines)

        bottom_lines = []
        for i in range(len(top_lines) - 1):
            bottom_lines.append(top_lines[i + 1])
        last_start_y = top_lines[-1]
        last_bottom = detect_bottom_edge(gray, start_y=last_start_y)
        bottom_lines.append(last_bottom)

        for i, (y1, y2) in enumerate(zip(top_lines, bottom_lines)):
            cropped = img_np[y1:y2, :]
            filename = f"receipt_page{page_idx+1}_{i+1}.{OUTPUT_FORMAT.lower()}"
            output_file = os.path.join(OUTPUT_DIR, filename)
            save_cropped_as_image(cropped, output_file, OUTPUT_FORMAT)
            receipt_count += 1
            print(f"  保存: {output_file} (高度: {y2 - y1}px)")

    print(f"\n✅ 完成！共分割出 {receipt_count} 个等高区域。")

if __name__ == "__main__":
    main()
