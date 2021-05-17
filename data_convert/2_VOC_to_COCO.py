import os
import sys
import argparse
import numpy as np

sys.path.append("../")

import utils.VOC_function

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--JPEGImages_path', help='The input dir of JPEGImages_path', type=str)
    parser.add_argument('--Annotations_path', help='The input dir of Annotations_path', type=str)
    parser.add_argument('--num_rate', help='The input dir of num_rate', type=float)
    parser.add_argument('--train_or_val', help='The input dir of train_or_val', type=str)
    args = parser.parse_args()

    # # ==============================================================================
    # TODO 输入
    classes_list = ['no_mask', 'have_mask']
    # python3 ./2_VOC_to_COCO.py --JPEGImages_path=./1_YOLO_to_VOC_data/train/JPEGImages/ --Annotations_path=./1_YOLO_to_VOC_data/train/Annotations/ --num_rate=1.0 --train_or_val=train
    # python3 ./2_VOC_to_COCO.py --JPEGImages_path=./1_YOLO_to_VOC_data/val/JPEGImages/ --Annotations_path=./1_YOLO_to_VOC_data/val/Annotations/ --num_rate=1.0 --train_or_val=val
    #
    #
    # classes_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #                 'A', 'B', 'C', 'D', 'E', 'F', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # python3 ./2_VOC_to_COCO.py --JPEGImages_path=../../../datasets/wzk/GEN/liuhao_voc_GEN/JPEGImages/ --Annotations_path=../../../datasets/wzk/GEN/liuhao_voc_GEN/Annotations/ --num_rate=0.02 --train_or_val=train
    #
    #
    # classes_list = ['sugarbeet', 'weed']
    # python3 ./2_VOC_to_COCO.py --JPEGImages_path=../../../../DeepLearning_course/深度学习与TensorFlow入门实战-源码和PPT_liuhao/lesson53-目标检测/yolov2-tf2_liuhao/data/train/image/ --Annotations_path=../../../../DeepLearning_course/深度学习与TensorFlow入门实战-源码和PPT_liuhao/lesson53-目标检测/yolov2-tf2_liuhao/data/train/image/ --Annotations_path=../../../../DeepLearning_course/深度学习与TensorFlow入门实战-源码和PPT_liuhao/lesson53-目标检测/yolov2-tf2_liuhao/data/train/annotation/ --num_rate=0.02 --train_or_val=train

    # ==============================================================================
    # TODO 输出保存在里面
    save_coco_path = "./2_VOC_to_COCO_data/{}/".format(args.train_or_val)

    ###==============================================================================
    classes_dict = dict()
    for key, item in enumerate(classes_list):
        classes_dict[item] = key
    print(classes_dict)  # {'no_mask': 0, 'have_mask': 1}

    ###==============================================================================

    img_name_list_all = np.sort(os.listdir(args.JPEGImages_path))
    img_name_list = img_name_list_all[0:int(args.num_rate * len(img_name_list_all))]
    print(len(img_name_list))
    print(img_name_list)

    ###==============================================================================
    # # TODO 根据name，寻找图片和标签

    json_dict = {"images": [],
                 "type": "instances",
                 "annotations": [],
                 "categories": []
                 }

    for cate, cid in classes_dict.items():
        cat = {'supercategory': 'none',
               'id': cid,
               'name': cate
               }
        json_dict['categories'].append(cat)

    bnd_id_SingleImage_all = 1  ###################### 这个要从1开始，因为ann的id从1开始
    for index in range(0, len(img_name_list), 1):  ### 这个要从0开始，因为image_id要从1开始
        img_name = img_name_list[index]
        name = img_name[img_name.rfind("/") + 1:img_name.rfind(".")]

        img_path = args.JPEGImages_path + img_name
        label_path = args.Annotations_path + "{}.xml".format(name)

        # # # TODO voc可视化，只需要这几个参数
        # utils.VOC_function.voc_visualize(img_path, label_path, name)

        # # # TODO voc转换为yolo，只需要这几个参数
        # # # TODO 注意：voc没有类别对应的数字，转化为yolo数据需要，指定类别对应的数字
        bnd_id_SingleImage_start = bnd_id_SingleImage_all
        bnd_id_SingleImage_end = utils.VOC_function.save_voc2coco(img_path, label_path, name, classes_dict,
                                                                  args.train_or_val, save_coco_path,
                                                                  index, bnd_id_SingleImage_start, json_dict)
        bnd_id_SingleImage_all = bnd_id_SingleImage_all + bnd_id_SingleImage_end

    ###==============================================================================
