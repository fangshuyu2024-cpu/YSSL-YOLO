import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from timm.models.layers import DropPath

def main():
    # 加载训练好的YOLO模型
    model = YOLO('runs/hole/32-111/1/weights/best.pt')  # 选择训练好的权重路径

    # 进行模型验证并输出交并比 (IoU) 和其他相关性能指标53

    results = model.val(
        data='dataset-hole/data.yaml',  # 数据集配置文件
        split='test',  # 验证数据集分区：train、val 或 test10
        imgsz=640,  # 输入图像大小3\3
        batch=17,  # 批量大小22
        project='runs/hole/32-111/test',  # 输出文件夹路径
        name='1',  # 实验名称
        save_json=True,  # 保存JSON文件以便计算COCO指标
        iou=0.5,  # 设置IoU阈值（可以调整，根据需求）
        agnostic_nms=True,  # 使用类无关的NMS
        save_txt=True,  # 保存每张图片的预测框信息
        verbose=True,  # 输出详细信息

    )

    # 打印 results_dict 内容，查看可用的键
    print("results_dict 内容：", results.results_dict)

    # 获取并打印IoU相关的指标
    # 通过解析验证结果，获取框和掩膜的IoU值
    if 'metrics/iou' in results.results_dict:
        print("交并比 (IoU):", results.results_dict['metrics/iou'])
    else:
        print("未找到IoU指标，可能需要查看其他指标或代码调整。")

    # 打印其他指标：精度、召回率、mAP等
    print("框的精度 (Box Precision):", results.results_dict.get('metrics/precision(B)', 'N/A'))
    print("框的召回率 (Box Recall):", results.results_dict.get('metrics/recall(B)', 'N/A'))
    print("框的 mAP@0.5 (Box mAP@0.5):", results.results_dict.get('metrics/mAP50(B)', 'N/A'))
    print("框的 mAP@0.5:0.95 (Box mAP@0.5:0.95):", results.results_dict.get('metrics/mAP50-95(B)', 'N/A'))
    print("掩膜的精度 (Mask Precision):", results.results_dict.get('metrics/precision(M)', 'N/A'))
    print("掩膜的召回率 (Mask Recall):", results.results_dict.get('metrics/recall(M)', 'N/A'))
    print("掩膜的 mAP@0.5 (Mask mAP@0.5):", results.results_dict.get('metrics/mAP50(M)', 'N/A'))
    print("掩膜的 mAP@0.5:0.95 (Mask mAP@0.5:0.95):", results.results_dict.get('metrics/mAP50-95(M)', 'N/A'))
    print("模型综合评分 (Fitness):", results.results_dict.get('fitness', 'N/A'))

# 只有当作为主程序执行时才启动多进程
if __name__ == '__main__':
    main()

