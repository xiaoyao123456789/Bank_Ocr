"D:\Yy\Documents\昆明出差报销\报销明细四\电路板维修\建设银行.pdf"

# split_receipts_by_logo.py
# 改进版：仅检测左侧区域的Logo，避免红色章误检

# split_receipts_equal_height.py
# 根据Logo最上边向上30mm切顶，最后块自动延展到底部文字

import fitz
import cv2
import numpy as np
from PIL import Image
import os
from sklearn.cluster import DBSCAN

# ================== 配置参数 ==================
PDF_PATH = r"D:\Yy\Documents\yuqing\6.pdf"         # ← 修改为你的PDF路径
OUTPUT_DIR = r"D:\Yy\Documents\yuqing"          # 输出目录

DPI = 300
TOP_OFFSET_MM = 10           # 从Logo最上方往上30mm开始切
MIN_LOGO_AREA = 150
CLUSTER_EPS = 15

LOGO_ROI_LEFT_RATIO = 0.0
LOGO_ROI_RIGHT_RATIO = 0.4
OUTPUT_FORMAT = "PNG"  # PNG or JPEG
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
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # ✅ 修复：正确赋值 mask_red
    mask_red = mask_red1 + mask_red2

    # 返回蓝 + 红
    return mask_blue + mask_red  # ✅ 现在没问题了
def find_logo_tops(img, dpi):
    """返回每个Logo最上方Y坐标（用于+30mm上移）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, w = img.shape[:2]

    x1 = int(w * LOGO_ROI_LEFT_RATIO)
    x2 = int(w * LOGO_ROI_RIGHT_RATIO)
    roi_hsv = hsv[:, x1:x2]
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
        logo_top_in_roi = y  # ROI内y坐标
        raw_tops.append(logo_top_in_roi)

    if not raw_tops:
        return []

    # 聚类去重（同一行）
    X = np.array(raw_tops).reshape(-1, 1)
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=1).fit(X).labels_
    unique_tops = [np.min(X[labels == label]) for label in set(labels)]  # 取每类最小Y（最上边）
    return sorted(unique_tops)

def detect_bottom_edge(img_gray, start_y, min_text_density=5):
    """从start_y向下扫描，找最后一行有文字的行"""
    h_proj = np.sum(img_gray < 200, axis=1)  # 黑色文字：灰度<200
    valid_rows = np.where(h_proj > min_text_density)[0]  # 有足够黑点的行
    if len(valid_rows) == 0:
        return img_gray.shape[0]  # 保守到底
    last_text_row = valid_rows[-1]
    return min(img_gray.shape[0], last_text_row + mm_to_px(10, DPI))  # 加10mm边距

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

        # 1. 找每个Logo最上边Y（在ROI中）
        logo_tops = find_logo_tops(img_np, DPI)
        if len(logo_tops) == 0:
            print(f"  ⚠️ 未检测到Logo，跳过")
            continue

        # 2. 计算每个回单的顶部切割线（上移30mm）
        top_lines = [max(0, top - px_offset) for top in logo_tops]
        top_lines = sorted(top_lines)

        # 3. 底部线：前N-1个用下一个top，最后一个特殊处理
        bottom_lines = []
        for i in range(len(top_lines) - 1):
            bottom_lines.append(top_lines[i + 1])
        # 最后一个：找底部文字 + 边距
        last_start_y = top_lines[-1]
        last_bottom = detect_bottom_edge(gray, start_y=last_start_y)
        bottom_lines.append(last_bottom)

        # 4. 裁剪并保存
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
