import os
import sys
import argparse
import numpy as np

sys.path.append("../")

import utils.YOLO_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_images', help='The input dir of yolo_images', type=str)
    parser.add_argument('--yolo_labels', help='The input dir of yolo_labels', type=str)
    parser.add_argument('--num_rate', help='The input dir of num_rate', type=float)
    parser.add_argument('--train_or_val', help='The input dir of train_or_val', type=str)
    args = parser.parse_args()

    ###==============================================================================
    ## TODO 输入
    classes_list = ['no_mask', 'have_mask']
    # python3 ./1_YOLO_to_VOC.py --yolo_images=../../datasets/kouzhao/new_mask_data/train/images/ --yolo_labels=../../datasets/kouzhao/new_mask_data/train/labels/ --num_rate=1.0 --train_or_val=train
    # python3 ./1_YOLO_to_VOC.py --yolo_images=../../datasets/kouzhao/new_mask_data/val/images/ --yolo_labels=../../datasets/kouzhao/new_mask_data/val/labels/ --num_rate=1.0 --train_or_val=val

    # =====================================================================
    ## TODO 输出保存在里面
    save_voc_dir = './1_YOLO_to_VOC_data/{}/'.format(args.train_or_val)

    ###==============================================================================
    classes_dict = dict()
    for key, item in enumerate(classes_list):
        classes_dict[key] = item
    print(classes_dict)  # {0: 'no_mask', 1: 'have_mask'}

    img_name_list_all = np.sort(os.listdir(args.yolo_images))
    img_name_list = img_name_list_all[0:int(args.num_rate * len(img_name_list_all))]
    print(len(img_name_list))
    print(img_name_list)

    ###==============================================================================
    # # # TODO 根据name，寻找图片和标签
    for index in range(0, len(img_name_list), 1):
        img_name = img_name_list[index]
        name = img_name[img_name.rfind("/") + 1:img_name.rfind(".")]

        img_path = args.yolo_images + img_name
        label_path = args.yolo_labels + "{}.txt".format(name)

        # # # # # TODO yolo可视化，只需要这几个参数
        # utils.YOLO_function.yolo_visualize(img_path, label_path, name, classes_dict)

        # TODO yolo转换为voc，只需要这几个参数
        utils.YOLO_function.save_yolo2voc(img_path, label_path, name, img_name, classes_dict, args.train_or_val, save_voc_dir)

    ###==============================================================================
