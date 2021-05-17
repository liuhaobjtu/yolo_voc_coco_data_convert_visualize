import os
import glob
import numpy as np
import json
import shutil
# import xml.etree.ElementTree as ET
import xml.etree.ElementTree
import cv2


def read_voc_xml(label_path):
    # ===============================================================
    # TODO xml文件名是没法获取图片后缀名的，必须解析xml的<filename>
    # ===============================================================

    """ Parse a PASCAL VOC xml file """
    tree = xml.etree.ElementTree.parse(source=label_path)
    # root = tree.getroot()

    img_name = tree.find('filename').text
    img_name = img_name[img_name.rfind("/") + 1:]
    # print("========================4:", img_name)

    size = tree.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    img_depth = int(size.find('depth').text)

    objects = []
    for index in range(0, len(tree.findall('object')), 1):
        obj_struct = {}
        obj_struct['name'] = tree.findall('object')[index].find('name').text
        bbox = tree.findall('object')[index].find('bndbox')

        obj_struct['bbox'] = [float(bbox.find('xmin').text),
                              float(bbox.find('ymin').text),
                              float(bbox.find('xmax').text),
                              float(bbox.find('ymax').text)]

        objects.append(obj_struct)

    return img_name, img_width, img_height, img_depth, objects


def voc_visualize(img_path, xml_path, img_name):
    xml_return_img_name, img_width, img_height, img_depth, objects = read_voc_xml(xml_path)

    # if len(objects) == 0:
    #     continue

    img_mat = cv2.imread(filename=img_path)

    # draw box and name
    for index in range(0, len(objects), 1):
        xmin, ymin, xmax, ymax = objects[index]['bbox']
        point1 = (int(xmin), int(ymin))  # TODO 重点是找到这四个点
        point2 = (int(xmax), int(ymin))
        point3 = (int(xmax), int(ymax))
        point4 = (int(xmin), int(ymax))

        class_name = objects[index]['name']
        print("标注：", class_name)

        # cv2.line(img=img_mat, pt1=point1, pt2=point3, color=(0, 0, 255), thickness=2)
        # cv2.rectangle(img=img_mat, pt1=point1, pt2=point3, color=(0, 0, 255), thickness=2)
        cv2.putText(img=img_mat, text=class_name, org=point3, fontFace=cv2.FONT_ITALIC, fontScale=2,
                    color=(0, 255, 0), thickness=5)

        cv2.circle(img=img_mat, center=point1, radius=1, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point2, radius=2, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point3, radius=3, color=(0, 255, 0), thickness=2)
        cv2.circle(img=img_mat, center=point4, radius=4, color=(0, 255, 0), thickness=2)

        point_list = [point1, point2, point3, point4]
        point_array = np.array(point_list, np.int32)
        # cv2.polylines(img=img_mat, pts=[point_array], isClosed=False, color=(0, 0, 255), thickness=2)  ##[point_array]！！！
        cv2.polylines(img=img_mat, pts=[point_array], isClosed=True, color=(0, 0, 255), thickness=2)  ##[point_array]！！！
        # cv2.fillPoly(img=img_mat, pts=[point_array], color=(0, 255, 0))

    # ===============================================================
    winname = img_name
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


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_voc2coco(img_path, label_path, name, classes_dict, train_or_val, save_path, index, bnd_id_SingleImage_start, json_dict):
    img_name, img_width, img_height, img_depth, objects = read_voc_xml(label_path)

    makedirs(save_path + "annotations/")

    flag_SingleImage = True
    annotation_dict = None
    json_dict_SingleImage = None
    if flag_SingleImage:
        makedirs(save_path + "annotations_{}2017/".format(train_or_val))

        # annotation_dict = {############ TODO 不能放在这里
        #     "coordinates": {
        #         "x": 0.0,
        #         "y": 0.0,
        #         "width": 0.0,
        #         "height": 0.0
        #     },
        #     "label": ""
        # }
        json_dict_SingleImage = [
            {"image": img_name,
             "annotations": []
             }
        ]

    # =====================================================================
    image_id = 1 + index
    image = {'file_name': img_name,
             'height': img_height,
             'width': img_width,
             'id': image_id
             }
    json_dict['images'].append(image)
    # =====================================================================
    count = 0
    for obj in objects:
        category = obj['name']

        category_id = classes_dict[category]
        # print("============:", category_id)

        xmin, ymin, xmax, ymax = obj['bbox']

        assert (xmax > xmin), "xmax <= xmin, {}".format(label_path)
        assert (ymax > ymin), "ymax <= ymin, {}".format(label_path)
        o_width = abs(xmax - xmin)  ################################# TODO 数据处理
        o_height = abs(ymax - ymin)  ################################ TODO 数据处理

        # print("========================:", bnd_id_SingleImage_start)
        ann = {'area': o_width * o_height,
               'iscrowd': 0,
               'image_id': image_id,
               'bbox': [xmin, ymin, o_width, o_height],
               'category_id': category_id,
               'id': bnd_id_SingleImage_start,
               'ignore': 0,
               'segmentation': []
               }
        json_dict['annotations'].append(ann)
        count = count + 1
        bnd_id_SingleImage_start = bnd_id_SingleImage_start + 1


        if flag_SingleImage:
            annotation_dict = {
                "coordinates": {
                    "x": 0.0,
                    "y": 0.0,
                    "width": 0.0,
                    "height": 0.0
                },
                "label": ""
            }

            annotation_dict["coordinates"]["x"] = xmin
            annotation_dict["coordinates"]["y"] = ymin
            annotation_dict["coordinates"]["width"] = o_width
            annotation_dict["coordinates"]["height"] = o_height
            annotation_dict["label"] = category

            json_dict_SingleImage[0]["annotations"].append(annotation_dict)

    # print(json_dict_SingleImage)
    # =====================================================================
    save_json = save_path + "annotations/" + 'instances_{}2017.json'.format(train_or_val)
    json_fp = open(file=save_json, mode='w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

    save_json_SingleImage = save_path + "annotations_{}2017/".format(train_or_val) + '{}.json'.format(name)
    json_fp_SingleImage = open(file=save_json_SingleImage, mode='w')
    json_str_SingleImage = json.dumps(json_dict_SingleImage)
    json_fp_SingleImage.write(json_str_SingleImage)
    json_fp_SingleImage.close()

    # =====================================================================
    dst = save_path + "/{}2017/".format(train_or_val)
    makedirs(dst)
    shutil.copy(src=img_path, dst=dst)  # copy会调用copyfile,但是copy的dst可以使目录
    # =====================================================================
    bnd_id_SingleImage_end = count
    return bnd_id_SingleImage_end

def save_voc2yolo(img_path, xml_path, name, classes_dict, save_yolo_path):

    xml_return_img_name, img_width, img_height, img_depth, objects = read_voc_xml(xml_path)
    # if len(objects) == 0:
    #     continue

    # 图片
    shutil.copy(src=img_path, dst=save_yolo_path + "images/")  # copy会调用copyfile,但是copy的dst可以使目录

    lines = []
    for index in range(0, len(objects), 1):
        xmin, ymin, xmax, ymax = objects[index]['bbox']
        class_name = objects[index]['name']
        label = classes_dict[class_name]
        # print("标注：", class_name, label)

        centerx = (xmax + xmin) / (2.0 * img_width)  ############### TODO 数据处理
        centery = (ymax + ymin) / (2.0 * img_height)  ############## TODO 数据处理
        width = (xmax - xmin) / (1.0 * img_width)  ################# TODO 数据处理
        height = (ymax - ymin) / (1.0 * img_height)  ################ TODO 数据处理
        if index < len(objects) - 1:
            line = "%s %.6f %.6f %.6f %.6f\n" % (label, centerx, centery, width, height)  # TODO 可以更改顺序
        else:
            line = "%s %.6f %.6f %.6f %.6f" % (label, centerx, centery, width, height)  # TODO 可以更改顺序
        # print(xml_path, line)
        lines.append(line)

    txt_name = save_yolo_path + "labels/" + name + ".txt"
    with open(file=txt_name, mode="w") as f:
        f.writelines(lines)
