

//TODO  20250915  18:58
//      添加中文注释
//      封装类

package com.ruoyi.web.controller.imageProcessing;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class BankReceiptDetector {

    private static final String PDF_PATH = "D:\\work\\数据版\\银行回单\\中行回单.pdf";
    private static final String OUTPUT_DIR = "D:\\work\\temp";
    private static final float DPI = 300f;
    private static final int TOP_OFFSET_MM = 10;
    private static final int MIN_LOGO_AREA = 10;
    private static final int CLUSTER_EPS = 15;
    private static final double LOGO_ROI_LEFT_RATIO = 0.0;
    private static final double LOGO_ROI_RIGHT_RATIO = 0.5;
    private static final String OUTPUT_FORMAT = "PNG";

    static {
        try {
            String opencvDll = System.getProperty("user.dir") + "/ruoyi-admin/lib/opencv_java452.dll";
            System.load(opencvDll);
            System.out.println("DLL 加载成功: " + opencvDll);
        } catch (Exception e) {
            throw new RuntimeException("OpenCV DLL 加载失败: " + e.getMessage(), e);
        }
    }

    public static void main(String[] args) {
        try {
            new BankReceiptDetector().process();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void process() throws IOException {
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) outputDir.mkdirs();

        PDDocument document = PDDocument.load(new File(PDF_PATH));
        PDFRenderer renderer = new PDFRenderer(document);
        int receiptCount = 0;
        int pxOffset = mmToPx(TOP_OFFSET_MM, DPI);

        for (int pageIdx = 0; pageIdx < document.getNumberOfPages(); pageIdx++) {
            System.out.println("正在处理第 " + (pageIdx + 1) + " / " + document.getNumberOfPages() + " 页...");
            BufferedImage bufImg = renderer.renderImageWithDPI(pageIdx, DPI, ImageType.RGB);
            Mat img = bufferedImageToMat(bufImg);
            Mat gray = new Mat();
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_RGB2GRAY);

            List<Integer> logoTops = findLogoTops(img);
            if (logoTops.isEmpty()) {
                System.out.println("  ⚠️ 未检测到Logo，跳过");
                continue;
            }

            System.out.println("  ✅ 检测到Logo位置: " + logoTops);

            List<Integer> topLines = new ArrayList<>();
            for (int top : logoTops) topLines.add(Math.max(0, top - pxOffset));
            Collections.sort(topLines);

            List<Integer> bottomLines = new ArrayList<>();
            for (int i = 0; i < topLines.size() - 1; i++) bottomLines.add(topLines.get(i + 1));
            int lastStartY = topLines.get(topLines.size() - 1);
            int lastBottom = detectBottomEdge(gray, lastStartY, img.cols() / 2);
            bottomLines.add(lastBottom);

            for (int i = 0; i < topLines.size(); i++) {
                int y1 = topLines.get(i);
                int y2 = bottomLines.get(i);
                Rect cropRect = new Rect(0, y1, img.cols(), y2 - y1);
                Mat cropped = new Mat(img, cropRect);

                int h = cropped.rows();
                int w = cropped.cols();
                if (h == 0 || w == 0) continue;

                double aspectRatio = (double) w / h;
                if (aspectRatio > 10 || aspectRatio < 0.1) continue;

                Mat saveMat = new Mat();
                Imgproc.cvtColor(cropped, saveMat, Imgproc.COLOR_RGB2BGR);

                String filename = String.format("receipt_page%d_%d.%s", pageIdx + 1, i + 1, OUTPUT_FORMAT.toLowerCase());
                String outputFile = new File(outputDir, filename).getAbsolutePath();
                Imgcodecs.imwrite(outputFile, saveMat);
                receiptCount++;
                System.out.printf("  保存: %s (起始Y=%d, 结束Y=%d, 高度=%d, 宽度=%d, ratio=%.2f)%n",
                        outputFile, y1, y2, y2 - y1, w, aspectRatio);
            }

        }

        document.close();
        System.out.println("\n✅ 完成！共分割出 " + receiptCount + " 个回单。");
    }

    private static int mmToPx(int mm, float dpi) {
        return (int) (dpi * mm / 25.4);
    }

    private Mat getColorMasks(Mat hsv) {
        // Python 完全一致的 HSV 范围
        Mat maskBlue = new Mat();
        Core.inRange(hsv, new Scalar(100, 50, 50), new Scalar(140, 255, 255), maskBlue);

        Mat maskRed1 = new Mat();
        Mat maskRed2 = new Mat();
        Core.inRange(hsv, new Scalar(0, 50, 50), new Scalar(10, 255, 255), maskRed1);
        Core.inRange(hsv, new Scalar(170, 50, 50), new Scalar(180, 255, 255), maskRed2);
        Mat maskRed = new Mat();
        Core.add(maskRed1, maskRed2, maskRed);

        Mat maskGreen = new Mat();
        Core.inRange(hsv, new Scalar(35, 50, 50), new Scalar(85, 255, 255), maskGreen);

        Mat maskYellow = new Mat();
        Core.inRange(hsv, new Scalar(20, 50, 50), new Scalar(35, 255, 255), maskYellow);

        Mat maskCustom = new Mat();
        Core.inRange(hsv, new Scalar(65, 90, 40), new Scalar(75, 255, 120), maskCustom);

        Mat mask = new Mat();
        Core.add(maskBlue, maskRed, mask);
        Core.add(mask, maskGreen, mask);
        Core.add(mask, maskYellow, mask);
        Core.add(mask, maskCustom, mask);

        // 形态学操作与 Python 保持一致
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);

        return mask;
    }

    private List<Integer> findLogoTops(Mat img) {
        int h = img.rows();
        int w = img.cols();
        int x1 = 0;
        int x2 = (int) (w * LOGO_ROI_RIGHT_RATIO);
        Mat roi = new Mat(img, new Rect(x1, 0, x2 - x1, h));
        Mat hsv = new Mat();
        Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_RGB2HSV);
        Mat mask = getColorMasks(hsv);
        if (Core.countNonZero(mask) == 0) {
            System.out.println("  ⚠️ Mask为空，未检测到颜色区域");
            return new ArrayList<>();
        }

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Integer> rawTops = new ArrayList<>();

        System.out.printf("  [Java] 图像尺寸: %dx%d, ROI: x1=%d, x2=%d%n", h, w, x1, x2);
        System.out.println("  [Java] Mask非零像素数量: " + Core.countNonZero(mask));
        System.out.println("  检测到轮廓数量: " + contours.size());
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            double area = Imgproc.contourArea(contour);
            if (area < MIN_LOGO_AREA) continue;

            Rect rect = Imgproc.boundingRect(contour);
            if (rect.x + rect.width > x2) continue; // 避免右侧杂色
            double aspectRatio = (double) rect.width / rect.height;
            if (aspectRatio > 20 || aspectRatio < 0.05) continue;

            rawTops.add(rect.y);
            System.out.printf("    轮廓%d: y=%d, area=%.2f, ratio=%.2f%n", i, rect.y, area, aspectRatio);
        }

        Collections.sort(rawTops);
        System.out.println("  聚类前 rawTops: " + rawTops);

        List<Integer> uniqueTops = new ArrayList<>();
        if (!rawTops.isEmpty()) uniqueTops.add(rawTops.get(0));
        for (int i = 1; i < rawTops.size(); i++) {
            if (Math.abs(rawTops.get(i) - uniqueTops.get(uniqueTops.size() - 1)) > CLUSTER_EPS)
                uniqueTops.add(rawTops.get(i));
        }

        System.out.println("  聚类后 uniqueTops: " + uniqueTops);
        return uniqueTops;
    }


    private int detectBottomEdge(Mat grayImg, int startY, int roiWidth) {
        int h = grayImg.rows();
        int margin = mmToPx(10, DPI);

        for (int y = h - 1; y >= startY; y--) {
            int count = 0;
            for (int x = 0; x < roiWidth; x++) { // 只扫描 ROI 左半部分
                if (grayImg.get(y, x)[0] < 200) count++;
            }
            if (count > 5) return Math.min(h, y + margin);
        }
        return h;
    }

    private Mat bufferedImageToMat(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        Mat mat = new Mat(h, w, CvType.CV_8UC3);
        byte[] data = new byte[w * h * 3];
        int idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                data[idx++] = (byte) ((rgb >> 16) & 0xFF);
                data[idx++] = (byte) ((rgb >> 8) & 0xFF);
                data[idx++] = (byte) (rgb & 0xFF);
            }
        }
        mat.put(0, 0, data);
        return mat;
    }
}
