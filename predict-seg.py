import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


def load_images_from_folder(folder):
    """加载文件夹中的所有图像"""
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))
    return images


def apply_mask(image, mask, color, alpha=0.5):
    """应用半透明掩码到图像上"""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def detect_and_segment(yolo_model, images, output_folder, class_colors=None):
    """执行检测和分割，并保存结果"""
    if class_colors is None:
        # 默认颜色映射 (BGR格式)
        class_colors = {
            0: (0, 0, 255),  # 类别0: 红色
            1: (0, 255, 0),  # 类别1: 绿色
            2: (255, 0, 0),  # 类别2: 蓝色
            # 可以添加更多类别颜色
        }

    for filename, img in images:
        # 创建原始图像的副本用于绘制结果
        result_img = img.copy()

        # 进行预测
        results = yolo_model(img)

        # # 获取边界框、类别和掩码
        # for result in results:
        #
        #     boxes = result.boxes
        #     masks = result.masks
        #
        #     if masks is not None:
        #         # 获取掩码数据
        #         masks = masks.data.cpu().numpy()
        #         for i, (box, mask) in enumerate(zip(boxes, masks)):
        #             # 处理边界框
        #             x1, y1, x2, y2 = map(int, box.xyxy[0])
        #             label = int(box.cls[0])
        #             conf = float(box.conf[0])
        #
        #             # 获取当前类别的颜色
        #             color = class_colors.get(label, (0, 255, 255))  # 默认使用青色
        #
        #             # 调整掩码大小以匹配原始图像
        #             mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        #             mask = (mask > 0.5).astype(np.uint8)  # 二值化掩码
        #
        #             # 应用掩码到图像
        #             result_img = apply_mask(result_img, mask, color)
        #
        #             # 绘制边界框
        #             cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        #
        #             # 添加标签和置信度
        #             label_text = f"{label} {conf:.2f}"
        #             cv2.putText(result_img, label_text, (x1, y1 - 10),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        #
        # # 保存结果图像
        # output_path = os.path.join(output_folder, filename)
        # cv2.imwrite(output_path, result_img)
        #没有预测框
        for filename, img in images:
            # 创建原始图像的副本用于绘制结果
            result_img = img.copy()

            # 进行预测
            results = yolo_model(img)

            for result in results:
                # 获取掩码
                masks = result.masks

                if masks is not None:
                    # 获取掩码数据
                    masks = masks.data.cpu().numpy()
                    boxes = result.boxes

                    for i, (box, mask) in enumerate(zip(boxes, masks)):
                        # 获取当前类别的颜色
                        label = int(box.cls[0])
                        color = class_colors.get(label, (0, 255, 255))  # 默认使用青色

                        # 调整掩码大小以匹配原始图像
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                        mask = (mask > 0.5).astype(np.uint8)  # 二值化掩码

                        # 应用掩码到图像
                        result_img = apply_mask(result_img, mask, color)

            # 保存结果图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result_img)


def main(input_folder, output_folder, model_path):
    """主函数"""
    # 加载YOLOv8分割模型
    model = YOLO(model_path)  # 确保这是训练好的分割模型

    # 加载输入图像
    images = load_images_from_folder(input_folder)

    # 确保输出文件夹存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 定义类别颜色映射 (根据你的数据集调整)
    class_colors = {
        0: (0, 0, 255),  # 类别0: 红色
        1: (0, 255, 0),  # 类别1: 绿色
        # 添加更多类别...
    }

    # 执行检测和分割
    detect_and_segment(model, images, output_folder, class_colors)


if __name__ == "__main__":
    # 配置路径
    input_folder = 'dataset-WBC/VOCdevkit/test/images'  # 输入图像文件夹
    output_folder = 'runs/dataset-WBC-32/segment'  # 输出结果文件夹
    model_path = 'runs/dataset-WBC-32/1/weights/best.pt'  # 训练好的分割模型

    # 运行主程序
    main(input_folder, output_folder, model_path)