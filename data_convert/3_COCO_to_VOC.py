import os
import sys
import argparse
import numpy as np

sys.path.append("../")

import utils.COCO_function

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='The input dir of img_dir', type=str)
    parser.add_argument('--json_file', help='The input dir of json_file', type=str)
    parser.add_argument('--num_rate', help='The input dir of num_rate', type=float)
    parser.add_argument('--train_or_val', help='The input dir of train_or_val', type=str)
    args = parser.parse_args()

    # =====================================================================================
    # # TODO 输入
    # python3 ./3_COCO_to_VOC.py --img_dir=./2_VOC_to_COCO_data/train/train2017/ --json_file=./2_VOC_to_COCO_data/train/annotations/instances_train2017.json --num_rate=1.0 --train_or_val=train
    # python3 ./3_COCO_to_VOC.py --img_dir=./2_VOC_to_COCO_data/val/val2017/ --json_file=./2_VOC_to_COCO_data/val/annotations/instances_val2017.json --num_rate=1.0 --train_or_val=val

    # =====================================================================================
    # TODO 输出保存在里面
    save_voc_dir = './3_COCO_to_VOC_data/{}/'.format(args.train_or_val)

    # =====================================================================================
    # TODO 根据json，寻找图片
    # TODO coco可视化，只需要这几个参数
    # utils.COCO_function.coco_visualize(args.img_dir, args.json_file, args.num_rate)

    utils.COCO_function.coco2voc(args.img_dir, args.json_file, args.num_rate, args.train_or_val, save_voc_dir)

    # =====================================================================================
