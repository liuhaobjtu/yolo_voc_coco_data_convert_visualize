import os
import sys
import argparse
import numpy as np
import json
import cv2


def read_coco_json(json_file):
    with open(file=json_file, mode='r') as file:
        json_data = json.load(fp=file)

        img_name_coco = json_data[0]["image"]
        annotations = json_data[0]["annotations"]

        # for index in range(0, len(annotations), 1):
        #     coordinates = annotations[index]["coordinates"]
        #     label = annotations[index]["label"]
        #     print(index + 1, coordinates, label)

    # # 返回的图片信息比较全
    return img_name_coco, annotations


def visualize(img_path, annotations):
    # 读取图片 imread(图片路径，读取模式)
    img_mat = cv2.imread(filename=img_path, flags=cv2.IMREAD_UNCHANGED)

    # print(img_mat.shape[0], img_mat.shape[1])

    # ==========================================================================
    for index in range(0, len(annotations), 1):
        coordinates = annotations[index]["coordinates"]
        label = annotations[index]["label"]
        # print(index + 1, coordinates, label)

        xmin = coordinates["x"]
        ymin = coordinates["y"]
        width = coordinates["width"]
        height = coordinates["height"]

        point1 = (int(xmin), int(ymin))  # TODO 重点是找到这四个点
        point2 = (int(xmin + width), int(ymin))
        point3 = (int(xmin + width), int(ymin + height))
        point4 = (int(xmin), int(ymin + height))

        # cv2.line(img=img_mat, pt1=point1, pt2=point3, color=(0, 0, 255), thickness=2)
        # cv2.rectangle(img=img_mat, pt1=point1, pt2=point3, color=(0, 0, 255), thickness=2)
        cv2.putText(img=img_mat, text=label, org=point3, fontFace=cv2.FONT_ITALIC, fontScale=2, color=(0, 255, 0),
                    thickness=5)

        cv2.circle(img=img_mat, center=point1, radius=1, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point2, radius=2, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point3, radius=3, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point4, radius=4, color=(0, 255, 0), thickness=2)

        point_list = [point1, point2, point3, point4]
        point_array = np.array(point_list, np.int32)
        # cv2.polylines(img=img_mat, pts=[point_array], isClosed=False, color=(0, 0, 255), thickness=2)  ##[point_array]！！！
        cv2.polylines(img=img_mat, pts=[point_array], isClosed=True, color=(0, 0, 255), thickness=2)  ##[point_array]！！！
        # cv2.fillPoly(img=img_mat, pts=[point_array], color=(0, 255, 0))

    # cv2.imwrite(filename="./img_mat.png", img=img_mat)

    # ==========================================================================

    winname = "img_mat"
    cv2.namedWindow(winname=winname, flags=cv2.WINDOW_NORMAL)
    cv2.moveWindow(winname=winname, x=0, y=0)  # 将显示窗口移到显示屏的相应位置
    cv2.resizeWindow(winname=winname, width=1600, height=900)
    cv2.imshow(winname=winname, mat=img_mat)

    # =================================================
    while True:
        key = cv2.waitKey(delay=10)
        # if (key >= 0):  #
        if (key == 27):  # 返回值:esc是27,
            print("获取的键值:", key)
            cv2.destroyWindow(winname=winname)
            break
        if (key == 13):  # 返回值:enter是13,提前结束for循环
            print("获取的键值:", key)
            cv2.destroyWindow(winname=winname)
            exit()

    # =================================================
    # key = cv2.waitKey(delay=0)
    # cv2.destroyWindow(winname=winname)

    # =================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='The input dir of img_dir', type=str)
    parser.add_argument('--json_file', help='The input dir of json_file', type=str)
    parser.add_argument('--num_rate', help='The input dir of num_rate', type=float)
    parser.add_argument('--train_or_val', help='The input dir of train_or_val', type=str)
    args = parser.parse_args()

    args.img_dir = "/home/liuhao/PycharmProjects/DeepLearning_liuhao/datasets/kouzhao/liuhao_coco_kouzhao/train/train2017/"
    args.json_file = "/home/liuhao/PycharmProjects/DeepLearning_liuhao/datasets/kouzhao/liuhao_coco_kouzhao/train/annotations_train2017/"
    args.num_rate = 0.02

    # =====================================================================================
    # TODO 根据json，寻找图片
    # TODO coco可视化，只需要这几个参数
    img_name_list_all = np.sort(os.listdir(args.img_dir))
    img_name_list = img_name_list_all[0:int(args.num_rate * len(img_name_list_all))]
    # print(len(img_name_list))
    # print(img_name_list)

    for index in range(0, len(img_name_list), 1):  ### 这个要从0开始，因为image_id要从1开始
        img_name = img_name_list[index]
        name = img_name[img_name.rfind("/") + 1:img_name.rfind(".")]

        img_path = args.img_dir + img_name
        label_path = args.json_file + "{}.json".format(name)

        img_name_coco, annotations = read_coco_json(label_path)

        if img_name == img_name_coco:
            visualize(img_path, annotations)

    # =====================================================================================
