import os
import shutil
import numpy as np
import cv2
# from lxml.etree import Element, SubElement, tostring
import lxml.etree


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def deal_yolo_txt(label_path, shape):
    with open(file=label_path) as file:
        data = file.readlines()

        classid_xmin_ymin_xmax_ymax = []
        for index in range(len(data)):
            line = data[index].strip().split(" ")

            classid = line[0]
            centerx = line[1]
            centery = line[2]
            width = line[3]
            height = line[4]

            # print("===================:", classid)
            # print("===================:", type(classid))
            ########################################################
            # TODO 数据处理2
            xmin = ((float(centerx) * 2 - float(width)) * (shape[1])) / 2
            ymin = ((float(centery) * 2 - float(height)) * (shape[0])) / 2

            xmax = ((float(centerx) * 2 + float(width)) * (shape[1])) / 2
            ymax = ((float(centery) * 2 + float(height)) * (shape[0])) / 2

            ########################################################

            classid_xmin_ymin_xmax_ymax.append([classid, xmin, ymin, xmax, ymax])

        return classid_xmin_ymin_xmax_ymax


def yolo_visualize(img_path, label_path, name, classes_dict):
    img_mat = cv2.imread(filename=img_path)
    classid_xmin_ymin_xmax_ymax = deal_yolo_txt(label_path, img_mat.shape)
    # print(classid_xmin_ymin_xmax_ymax)

    for classid, xmin, ymin, xmax, ymax in classid_xmin_ymin_xmax_ymax:
        point1 = (int(xmin), int(ymin))  # TODO 重点是找到这四个点
        point2 = (int(xmax), int(ymin))
        point3 = (int(xmax), int(ymax))
        point4 = (int(xmin), int(ymax))

        # cv2.line(img=img_mat, pt1=point1, pt2=point3, color=(0, 0, 255), thickness=2)
        # cv2.rectangle(img=img_mat, pt1=point1, pt2=point3, color=(0, 0, 255), thickness=2)
        cv2.putText(img=img_mat, text=classes_dict[int(classid)], org=point3, fontFace=cv2.FONT_ITALIC, fontScale=2,
                    color=(0, 255, 0), thickness=5)

        cv2.circle(img=img_mat, center=point1, radius=1, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point2, radius=2, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point3, radius=3, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point4, radius=4, color=(0, 255, 0), thickness=2)

        point_list = [point1, point2, point3, point4]
        point_array = np.array(point_list, np.int32)
        # cv2.polylines(img=img_mat, pts=[point_array], isClosed=False, color=(0, 0, 255), thickness=2)  ##[point_array]！！！
        cv2.polylines(img=img_mat, pts=[point_array], isClosed=True, color=(0, 0, 255),
                      thickness=2)  ##[point_array]！！！
        # cv2.fillPoly(img=img_mat, pts=[point_array], color=(0, 255, 0))

    winname = name
    cv2.namedWindow(winname=winname, flags=cv2.WINDOW_NORMAL)
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


def save_yolo2voc(img_path, label_path, name, img_name, classes_dict, train_or_val, save_voc_dir):
    makedirs(save_voc_dir + "JPEGImages/")
    makedirs(save_voc_dir + "Annotations/")
    makedirs(save_voc_dir + "ImageSets/")

    img_mat = cv2.imread(filename=img_path)
    classid_xmin_ymin_xmax_ymax = deal_yolo_txt(label_path, img_mat.shape)
    # print(classid_xmin_ymin_xmax_ymax)

    makedirs(save_voc_dir + "ImageSets/" + "Main/")
    with open(save_voc_dir + "ImageSets/" + "Main/" + "{}.txt".format(train_or_val), "a") as file:
        file.write(name)
        file.write("\n")

    shutil.copy(src=img_path, dst=save_voc_dir + "JPEGImages/")  # copy会调用copyfile,但是copy的dst可以使目录

    node_root = lxml.etree.Element('annotation')
    node_folder = lxml.etree.SubElement(node_root, 'folder')
    node_folder.text = "VOC2012"
    node_filename = lxml.etree.SubElement(node_root, 'filename')
    node_filename.text = img_name

    node_path = lxml.etree.SubElement(node_root, 'path')
    node_path.text = os.path.abspath(img_path)    ########## TODO 返回图片全路径，这里可以控制xml文件里面的path

    node_size = lxml.etree.SubElement(node_root, 'size')
    node_width = lxml.etree.SubElement(node_size, 'width')
    node_width.text = str(img_mat.shape[1])
    # print("======================================:", node_width.text)
    # print("======================================:", type(node_width.text))
    # print("======================================:", type(int(node_width.text)))

    node_height = lxml.etree.SubElement(node_size, 'height')
    node_height.text = str(img_mat.shape[0])
    # print("======================================:", node_height.text)
    # print("======================================:", type(node_height.text))
    # print("======================================:", type(int(node_height.text)))

    node_depth = lxml.etree.SubElement(node_size, 'depth')
    node_depth.text = '3'

    for classid, xmin, ymin, xmax, ymax in classid_xmin_ymin_xmax_ymax:
        node_object = lxml.etree.SubElement(node_root, 'object')
        node_name = lxml.etree.SubElement(node_object, 'name')

        # 这里要根据classid分开处理
        node_name.text = classes_dict[int(classid)]

        node_difficult = lxml.etree.SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = lxml.etree.SubElement(node_object, 'bndbox')
        node_xmin = lxml.etree.SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(xmin))
        node_ymin = lxml.etree.SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(ymin))
        node_xmax = lxml.etree.SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(xmax))
        node_ymax = lxml.etree.SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(ymax))

    xml = lxml.etree.tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    xml_name = name + ".xml"
    # print(xml_name)

    with open(file=save_voc_dir + "Annotations/" + xml_name, mode="wb") as f:
        f.write(xml)
        f.close()
