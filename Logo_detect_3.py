# 相较于第一版做了以下优化
# ROI 只在左侧 1/3，HSV 转换区域更小 → 提速。
#
# 相邻聚类 代替 DBSCAN，不依赖 sklearn，逻辑也更快。


###可执行
######## #建行
         #兰州
         #工行
         #邮政

###待优化
####### #甘肃银行    -->中间logo 黄色
        #农行回单    -->聚类面积小        -> 50阈值正常
        #中行回单    -->中行回单         -> 加了区域长宽比过滤机制正常


import fitz
import cv2
import numpy as np
from PIL import Image
import os

# ================== 配置参数 ==================
# PDF_PATH = r"D:\Yy\Documents\yuqing\甘肃银行.pdf"
# PDF_PATH = r"D:\Yy\Documents\yuqing\工行回单1.pdf"
# PDF_PATH = r"D:\Yy\Documents\yuqing\建设银行.pdf"
# PDF_PATH = r"D:\Yy\Documents\yuqing\兰行回单.pdf"
# PDF_PATH = r"D:\Yy\Documents\yuqing\农行回单.pdf"
# PDF_PATH = r"D:\Yy\Documents\yuqing\邮储回单.pdf"
PDF_PATH = r"D:\Yy\Documents\yuqing\中行回单.pdf"

# OUTPUT_DIR = r"D:\Yy\Documents\yuqing\甘肃银行"
# OUTPUT_DIR = r"D:\Yy\Documents\yuqing\工行回单1"
# OUTPUT_DIR = r"D:\Yy\Documents\yuqing\建设银行"
# OUTPUT_DIR = r"D:\Yy\Documents\yuqing\兰行回单"
# OUTPUT_DIR = r"D:\Yy\Documents\yuqing\甘肃银行"
# OUTPUT_DIR = r"D:\Yy\Documents\yuqing\农行回单"
# OUTPUT_DIR = r"D:\Yy\Documents\yuqing\邮储回单"
OUTPUT_DIR = r"D:\Yy\Documents\yuqing\中行回单"

DPI = 300
TOP_OFFSET_MM = 10
MIN_LOGO_AREA = 50
CLUSTER_EPS = 15  # 聚类的容忍距离(px)

LOGO_ROI_LEFT_RATIO = 0.0
LOGO_ROI_RIGHT_RATIO = 1.0 / 2.0   # ← 左侧三分之一
OUTPUT_FORMAT = "PNG"
# =============================================

def mm_to_px(mm, dpi):
    return int(dpi * mm / 25.4)

def get_color_masks(hsv):
    """提取蓝色、红色、绿色、黄色和指定颜色(#009882)区域的掩码"""
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

    # 绿色
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 黄色
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 单独指定颜色 #009882 -> HSV大约 (69, 100%, 60%)
    lower_custom = np.array([65, 90, 40])
    upper_custom = np.array([75, 255, 120])
    mask_custom = cv2.inRange(hsv, lower_custom, upper_custom)

    # 合并
    return mask_blue + mask_red + mask_green + mask_yellow + mask_custom



def find_logo_tops(img, dpi):
    """返回每个Logo最上方Y坐标"""
    h, w = img.shape[:2]
    x1 = int(w * LOGO_ROI_LEFT_RATIO)
    x2 = int(w * LOGO_ROI_RIGHT_RATIO)

    roi_hsv = cv2.cvtColor(img[:, x1:x2], cv2.COLOR_RGB2HSV)
    mask = get_color_masks(roi_hsv)

    if cv2.countNonZero(mask) == 0:
        print("  ⚠️ ROI内没有检测到指定颜色像素（可能颜色范围不匹配）")
        return []

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  ⚠️ 颜色区域存在，但未形成有效轮廓（可能噪点太碎）")
        return []

    raw_tops = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.2:  # 太细长的去掉
            continue

        if area < MIN_LOGO_AREA:
            continue
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        aspect_ratio = w_cnt / float(h_cnt)
        if aspect_ratio > 5 or aspect_ratio < 0.2:  # 太细长的去掉
            continue
        raw_tops.append(y)

    if not raw_tops:
        print(f"  ⚠️ 检测到的轮廓面积都小于阈值 MIN_LOGO_AREA={MIN_LOGO_AREA}")
        return []

    if not raw_tops:
        return []

    # ==== 相邻聚类替代 DBSCAN ====
    raw_tops.sort()
    unique_tops = [raw_tops[0]]
    for t in raw_tops[1:]:
        if abs(t - unique_tops[-1]) > CLUSTER_EPS:
            unique_tops.append(t)

    if not unique_tops:
        print("  ⚠️ 所有候选Logo位置在聚类后被合并或过滤掉")
        return []

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
            h_cropped, w_cropped = cropped.shape[:2]

            if h_cropped == 0 or w_cropped == 0:
                print(f"  ⚠️ 裁剪区域为空，跳过 (y1={y1}, y2={y2})")
                continue

            aspect_ratio = w_cropped / float(h_cropped)

            if aspect_ratio > 10 or aspect_ratio < 0.1:
                print(f"  ⚠️ 裁剪区域长宽比异常 (w={w_cropped}, h={h_cropped}, ratio={aspect_ratio:.2f})，跳过")
                continue

            filename = f"receipt_page{page_idx + 1}_{i + 1}.{OUTPUT_FORMAT.lower()}"
            output_file = os.path.join(OUTPUT_DIR, filename)
            save_cropped_as_image(cropped, output_file, OUTPUT_FORMAT)
            receipt_count += 1
            print(f"  保存: {output_file} (高度: {y2 - y1}px, 宽度: {w_cropped}px)")

    print(f"\n✅ 完成！共分割出 {receipt_count} 个等高区域。")

if __name__ == "__main__":
    main()
