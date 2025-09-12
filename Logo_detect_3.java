#兼容所有情况的java代码
package com.bankdetect;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class BankReceiptDetector {
    
    // 配置参数
    private static final String PDF_PATH = "D:\\Yy\\Documents\\yuqing\\中行回单.pdf";
    private static final String OUTPUT_DIR = "D:\\Yy\\Documents\\yuqing\\中行回单";
    private static final float DPI = 300f;
    private static final int TOP_OFFSET_MM = 10;
    private static final int MIN_LOGO_AREA = 20;
    private static final int CLUSTER_EPS = 15;
    private static final double LOGO_ROI_LEFT_RATIO = 0.0;
    private static final double LOGO_ROI_RIGHT_RATIO = 1.0 / 2.0;
    private static final String OUTPUT_FORMAT = "PNG";
    
    static {
        // 加载OpenCV本地库
        nu.pattern.OpenCV.loadLocally();
    }
    
    public static void main(String[] args) {
        try {
            new BankReceiptDetector().process();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void process() throws IOException {
        // 创建输出目录
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        
        PDDocument document = PDDocument.load(new File(PDF_PATH));
        PDFRenderer pdfRenderer = new PDFRenderer(document);
        int receiptCount = 0;
        int pxOffset = mmToPx(TOP_OFFSET_MM, DPI);
        
        for (int pageIdx = 0; pageIdx < document.getNumberOfPages(); pageIdx++) {
            System.out.println("正在处理第 " + (pageIdx + 1) + " / " + document.getNumberOfPages() + " 页...");
            
            // 渲染PDF页面为图像
            BufferedImage bufferedImage = pdfRenderer.renderImageWithDPI(pageIdx, DPI, ImageType.RGB);
            Mat imgMat = bufferedImageToMat(bufferedImage);
            Mat grayMat = new Mat();
            Imgproc.cvtColor(imgMat, grayMat, Imgproc.COLOR_RGB2GRAY);
            
            // 检测Logo位置
            List<Integer> logoTops = findLogoTops(imgMat, DPI);
            if (logoTops.isEmpty()) {
                System.out.println("  ⚠️ 未检测到Logo，跳过");
                continue;
            }
            
            System.out.println("  ✅ 检测到 " + logoTops.size() + " 个Logo位置: " + logoTops);
            
            // 计算分割线
            List<Integer> topLines = new ArrayList<>();
            for (int top : logoTops) {
                topLines.add(Math.max(0, top - pxOffset));
            }
            Collections.sort(topLines);
            
            List<Integer> bottomLines = new ArrayList<>();
            for (int i = 0; i < topLines.size() - 1; i++) {
                bottomLines.add(topLines.get(i + 1));
            }
            int lastStartY = topLines.get(topLines.size() - 1);
            int lastBottom = detectBottomEdge(grayMat, lastStartY);
            bottomLines.add(lastBottom);
            
            // 裁剪并保存
            for (int i = 0; i < topLines.size(); i++) {
                int y1 = topLines.get(i);
                int y2 = bottomLines.get(i);
                
                Rect cropRect = new Rect(0, y1, imgMat.cols(), y2 - y1);
                Mat cropped = new Mat(imgMat, cropRect);
                
                int hCropped = cropped.rows();
                int wCropped = cropped.cols();
                
                if (hCropped == 0 || wCropped == 0) {
                    System.out.println("  ⚠️ 裁剪区域为空，跳过 (y1=" + y1 + ", y2=" + y2 + ")");
                    continue;
                }
                
                double aspectRatio = (double) wCropped / hCropped;
                
                if (aspectRatio > 10 || aspectRatio < 0.1) {
                    System.out.printf("  ⚠️ 裁剪区域长宽比异常 (w=%d, h=%d, ratio=%.2f)，跳过%n", wCropped, hCropped, aspectRatio);
                    continue;
                }
                
                // 转换为BGR格式用于保存
                Mat croppedBGR = new Mat();
                Imgproc.cvtColor(cropped, croppedBGR, Imgproc.COLOR_RGB2BGR);
                
                String filename = String.format("receipt_page%d_%d.%s", 
                    pageIdx + 1, i + 1, OUTPUT_FORMAT.toLowerCase());
                String outputFile = new File(outputDir, filename).getAbsolutePath();
                
                Imgcodecs.imwrite(outputFile, croppedBGR);
                receiptCount++;
                System.out.printf("  保存: %s (高度: %dpx, 宽度: %dpx)%n", outputFile, y2 - y1, wCropped);
            }
        }
        
        document.close();
        System.out.println("\n✅ 完成！共分割出 " + receiptCount + " 个等高区域。");
    }
    
    private static int mmToPx(int mm, float dpi) {
        return (int) (dpi * mm / 25.4);
    }
    
    private Mat getColorMasks(Mat hsv) {
        // 蓝色范围 - 扩大范围
        Scalar lowerBlue = new Scalar(90, 30, 30);
        Scalar upperBlue = new Scalar(150, 255, 255);
        Mat maskBlue = new Mat();
        Core.inRange(hsv, lowerBlue, upperBlue, maskBlue);
        
        // 红色范围（跨0度）- 扩大范围
        Scalar lowerRed1 = new Scalar(0, 30, 30);
        Scalar upperRed1 = new Scalar(15, 255, 255);
        Scalar lowerRed2 = new Scalar(165, 30, 30);
        Scalar upperRed2 = new Scalar(180, 255, 255);
        
        Mat maskRed1 = new Mat();
        Mat maskRed2 = new Mat();
        Core.inRange(hsv, lowerRed1, upperRed1, maskRed1);
        Core.inRange(hsv, lowerRed2, upperRed2, maskRed2);
        
        Mat maskRed = new Mat();
        Core.add(maskRed1, maskRed2, maskRed);
        
        // 绿色范围 - 扩大范围
        Scalar lowerGreen = new Scalar(30, 30, 30);
        Scalar upperGreen = new Scalar(90, 255, 255);
        Mat maskGreen = new Mat();
        Core.inRange(hsv, lowerGreen, upperGreen, maskGreen);
        
        // 黄色范围 - 扩大范围
        Scalar lowerYellow = new Scalar(15, 30, 30);
        Scalar upperYellow = new Scalar(40, 255, 255);
        Mat maskYellow = new Mat();
        Core.inRange(hsv, lowerYellow, upperYellow, maskYellow);
        
        // 自定义颜色 #009882 -> HSV大约 (69, 100%, 60%) - 扩大范围
        Scalar lowerCustom = new Scalar(60, 50, 20);
        Scalar upperCustom = new Scalar(80, 255, 150);
        Mat maskCustom = new Mat();
        Core.inRange(hsv, lowerCustom, upperCustom, maskCustom);
        
        // 紫色/品红色范围 - 新增
        Scalar lowerPurple = new Scalar(120, 30, 30);
        Scalar upperPurple = new Scalar(170, 255, 255);
        Mat maskPurple = new Mat();
        Core.inRange(hsv, lowerPurple, upperPurple, maskPurple);
        
        // 合并所有掩码
        Mat result = new Mat();
        Core.add(maskBlue, maskRed, result);
        Core.add(result, maskGreen, result);
        Core.add(result, maskYellow, result);
        Core.add(result, maskCustom, result);
        Core.add(result, maskPurple, result);
        
        return result;
    }
    
    private List<Integer> findLogoTops(Mat img, float dpi) {
        int h = img.rows();
        int w = img.cols();
        int x1 = (int) (w * LOGO_ROI_LEFT_RATIO);
        int x2 = (int) (w * LOGO_ROI_RIGHT_RATIO);
        
        // 提取ROI区域
        Rect roiRect = new Rect(x1, 0, x2 - x1, h);
        Mat roi = new Mat(img, roiRect);
        
        // 转换为HSV
        Mat roiHsv = new Mat();
        Imgproc.cvtColor(roi, roiHsv, Imgproc.COLOR_RGB2HSV);
        
        // 获取颜色掩码
        Mat mask = getColorMasks(roiHsv);
        
        // 检查是否有颜色像素
        if (Core.countNonZero(mask) == 0) {
            System.out.println("  ⚠️ ROI内没有检测到指定颜色像素（可能颜色范围不匹配）");
            return new ArrayList<>();
        }
        
        // 形态学操作
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        
        // 查找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        
        if (contours.isEmpty()) {
            System.out.println("  ⚠️ 颜色区域存在，但未形成有效轮廓（可能噪点太碎）");
            return new ArrayList<>();
        }
        
        List<Integer> rawTops = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            
            // 只保留最基本的面积过滤，放宽其他条件
            if (area < MIN_LOGO_AREA) {
                continue;
            }
            
            Rect boundingRect = Imgproc.boundingRect(contour);
            
            // 放宽长宽比限制，允许更多形状
            double aspectRatio = (double) boundingRect.width / boundingRect.height;
            if (aspectRatio > 20 || aspectRatio < 0.05) { // 只过滤极端异常的形状
                continue;
            }
            
            rawTops.add(boundingRect.y);
            System.out.println("  检测到Logo候选: y=" + boundingRect.y + ", 面积=" + (int)area + ", 长宽比=" + String.format("%.2f", aspectRatio));
        }
        
        if (rawTops.isEmpty()) {
            System.out.println("  ⚠️ 检测到的轮廓面积都小于阈值 MIN_LOGO_AREA=" + MIN_LOGO_AREA);
            return new ArrayList<>();
        }
        
        // 相邻聚类
        Collections.sort(rawTops);
        List<Integer> uniqueTops = new ArrayList<>();
        uniqueTops.add(rawTops.get(0));
        
        for (int i = 1; i < rawTops.size(); i++) {
            int current = rawTops.get(i);
            int last = uniqueTops.get(uniqueTops.size() - 1);
            if (Math.abs(current - last) > CLUSTER_EPS) {
                uniqueTops.add(current);
            }
        }
        
        if (uniqueTops.isEmpty()) {
            System.out.println("  ⚠️ 所有候选Logo位置在聚类后被合并或过滤掉");
            return new ArrayList<>();
        }
        
        return uniqueTops;
    }
    
    private int detectBottomEdge(Mat grayImg, int startY) {
        int height = grayImg.rows();
        int width = grayImg.cols();
        int minTextDensity = 5;
        
        // 计算水平投影（每行非白色像素数量）
        int[] hProj = new int[height];
        for (int y = 0; y < height; y++) {
            int count = 0;
            for (int x = 0; x < width; x++) {
                double[] pixel = grayImg.get(y, x);
                if (pixel[0] < 200) { // 非白色像素
                    count++;
                }
            }
            hProj[y] = count;
        }
        
        // 找到有文本的行
        List<Integer> validRows = new ArrayList<>();
        for (int y = 0; y < height; y++) {
            if (hProj[y] > minTextDensity) {
                validRows.add(y);
            }
        }
        
        if (validRows.isEmpty()) {
            return height;
        }
        
        // 获取最后一行有文本的位置
        int lastTextRow = validRows.get(validRows.size() - 1);
        
        // 添加10mm的边距
        int margin = mmToPx(10, DPI);
        return Math.min(height, lastTextRow + margin);
    }
    
    private Mat bufferedImageToMat(BufferedImage bufferedImage) {
        byte[] pixels = new byte[bufferedImage.getWidth() * bufferedImage.getHeight() * 3];
        int index = 0;
        
        for (int y = 0; y < bufferedImage.getHeight(); y++) {
            for (int x = 0; x < bufferedImage.getWidth(); x++) {
                int rgb = bufferedImage.getRGB(x, y);
                pixels[index++] = (byte) ((rgb >> 16) & 0xFF); // R
                pixels[index++] = (byte) ((rgb >> 8) & 0xFF);  // G
                pixels[index++] = (byte) (rgb & 0xFF);         // B
            }
        }
        
        Mat mat = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, pixels);
        return mat;
    }
}
