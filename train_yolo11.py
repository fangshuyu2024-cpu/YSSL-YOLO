import sys
import argparse
import os

# https://www.codewithgpu.com/u/iscyy

# sys.path.append('/root/ultralyticsPro/') # Path 以Autodl为例

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) # 直接加载yaml文件训练
    # model = YOLO(weights)  # 直接加载权重文件进行训练
    # model = YOLO(yaml).load(weights) # 加载yaml配置文件的同时，加载权重进行训练

    model.info()

    results = model.train(data='dataset-hole/data.yaml',  # 训练参数均可以重新设置
                          cache=False,
                          epochs=300,
                          imgsz=640,
                          workers=4,
                          batch=32,
                          close_mosaic=0,
                          project='runs/hole/32-111',
                          name='1',
                          optimizer='SGD'
                        # ...在这里添加需要修改的参数
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='self/32-111.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
#

# import warnings
#
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO(r'E:\4_fangshuyu\chengxu2\ultralytics-YOLO11\self\33.yaml')
#     model.train(data=r'E:\4_fangshuyu\chengxu2\ultralytics-YOLO11\dataset\data.yaml',
#                 cache=False,
#                 imgsz=640,
#                 epochs=300,
#                 single_cls=False,  # 是否是单类别检测
#                 batch=32,
#                 close_mosaic=0,
#                 workers=4,
#                 device='0',
#                 optimizer='SGD',
#                 amp=True,
#                 project='runs/train',
#                 name='33',
#                 )

