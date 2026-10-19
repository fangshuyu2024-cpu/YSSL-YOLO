import os
import cv2
import torch
from pathlib import Path



def run_yolov11_seg_detection(weights_path, source_dir, output_dir, conf_thres=0.25, iou_thres=0.45):
    """
    使用YOLOv11-Seg模型进行批量图片检测

    参数:
        weights_path: best.pt权重文件路径
        source_dir: 待检测图片目录
        output_dir: 检测结果输出目录
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('WongKinYiu/yolov11', 'custom', path=weights_path, source='github')
    # model = torch.hub.load('local_path_to_yolov11', 'custom', path='path_to_your_weights_file', source='local')

    model.to(device)
    model.conf = conf_thres  # 置信度阈值
    model.iou = iou_thres  # NMS IOU阈值

    # 获取图片列表
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(source_dir).glob(ext))

    print(f"Found {len(image_files)} images to process")

    # 批量处理图片
    for image_path in image_files:
        # 读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error loading image: {image_path}")
            continue

        # 推理
        results = model(img)

        # 渲染结果
        rendered_img = results.render()[0]

        # 保存结果
        output_path = os.path.join(output_dir, f"{image_path.stem}_result{image_path.suffix}")
        cv2.imwrite(output_path, rendered_img)
        print(f"Saved result to: {output_path}")

    print("Batch detection completed!")
    for result in results:
        print("Detections:")
        print(result.pandas().xyxy[0])  # 打印检测结果表格

if __name__ == "__main__":
    # 参数配置
    weights_path = "runs/dataset-hole-32/1/weights/best.pt"  # 替换为你的best.pt路径
    source_dir = "dataset-hole/images/test"  # 待检测图片目录
    output_dir = "runs/dataset-hole-32/detect/"  # 结果输出目录

    # 运行检测
    run_yolov11_seg_detection(
        weights_path=weights_path,
        source_dir=source_dir,
        output_dir=output_dir,
        conf_thres=0.25,
        iou_thres=0.45
    )