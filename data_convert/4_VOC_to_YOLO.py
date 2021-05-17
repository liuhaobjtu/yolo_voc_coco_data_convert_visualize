import os
import sys
import argparse
import numpy as np

sys.path.append("../")

import utils.VOC_function

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--JPEGImages_path', help='The input dir of JPEGImages_path', type=str)
    parser.add_argument('--Annotations_path', help='The input dir of Annotations_path', type=str)
    parser.add_argument('--num_rate', help='The input dir of num_rate', type=float)
    parser.add_argument('--train_or_val', help='The input dir of train_or_val', type=str)
    args = parser.parse_args()

    # # ==============================================================================
    ## TODO 输入
    classes_list = ['no_mask', 'have_mask']
    # python3 ./4_VOC_to_YOLO.py --JPEGImages_path=./3_COCO_to_VOC_data/train/JPEGImages/ --Annotations_path=./3_COCO_to_VOC_data/train/Annotations/ --num_rate=1.0 --train_or_val=train
    # python3 ./4_VOC_to_YOLO.py --JPEGImages_path=./3_COCO_to_VOC_data/val/JPEGImages/ --Annotations_path=./3_COCO_to_VOC_data/val/Annotations/ --num_rate=1.0 --train_or_val=val
    #
    #
    # classes_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #                 'A', 'B', 'C', 'D', 'E', 'F', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # python3 ./4_VOC_to_YOLO.py --JPEGImages_path=../../../datasets/wzk/GEN/liuhao_voc_GEN/JPEGImages/ --Annotations_path=../../../datasets/wzk/GEN/liuhao_voc_GEN/Annotations/ --num_rate=0.02 --train_or_val=train
    #
    #
    # classes_list = ['sugarbeet', 'weed']
    # python3 ./4_VOC_to_YOLO.py --JPEGImages_path=../../../../DeepLearning_course/深度学习与TensorFlow入门实战-源码和PPT_liuhao/lesson53-目标检测/yolov2-tf2_liuhao/data/train/image/ --Annotations_path=../../../../DeepLearning_course/深度学习与TensorFlow入门实战-源码和PPT_liuhao/lesson53-目标检测/yolov2-tf2_liuhao/data/train/annotation/ --num_rate=0.02 --train_or_val=train

    # ==============================================================================
    ## TODO 输出保存在里面
    save_yolo_path = "./4_VOC_to_YOLO_data/{}/".format(args.train_or_val)

    ###==============================================================================
    classes_dict = dict()
    for key, item in enumerate(classes_list):
        classes_dict[item] = key
    print(classes_dict)  # {'no_mask': 0, 'have_mask': 1}

    img_name_list_all = np.sort(os.listdir(args.JPEGImages_path))
    img_name_list = img_name_list_all[0:int(args.num_rate * len(img_name_list_all))]
    print(img_name_list)

    ###==============================================================================
    makedirs(save_yolo_path + "labels/")
    makedirs(save_yolo_path + "images/")

    with open(file=save_yolo_path + "classes.txt", mode="a") as f:
        for index, key in enumerate(classes_dict.keys()):
            # print(index, key)
            f.writelines(key)
            f.writelines("\n")

    # # TODO 根据name，寻找图片和标签
    for index in range(0, len(img_name_list), 1):
        img_name = img_name_list[index]
        name = img_name[img_name.rfind("/") + 1:img_name.rfind(".")]

        img_path = args.JPEGImages_path + img_name
        label_path = args.Annotations_path + "{}.xml".format(name)

        # # # TODO voc可视化，只需要这几个参数
        # utils.VOC_function.voc_visualize(img_path, label_path, name)

        # TODO voc转换为yolo，只需要这几个参数
        # # TODO 注意：voc没有类别对应的数字，转化为yolo数据需要，指定类别对应的数字
        utils.VOC_function.save_voc2yolo(img_path, label_path, name, classes_dict, save_yolo_path)

    ###==============================================================================
