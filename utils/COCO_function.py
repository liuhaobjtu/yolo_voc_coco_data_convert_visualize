import os
import json
import shutil
import numpy as np
import cv2


# from pycocotools.coco import COCO


def read_coco_json(json_data, index):
    img_name = json_data["images"][index]["file_name"]  # TODO 根据index来遍历图片
    img_height = json_data["images"][index]["height"]  ## TODO 根据index来遍历图片
    img_width = json_data["images"][index]["width"]  #### TODO 根据index来遍历图片
    id = json_data["images"][index]["id"]  ############## TODO 根据index来遍历图片
    print("图片img_name：", img_name)

    # TODO 找到图片标注
    # # #指定类别
    # if json_data["annotations"][i]['category_id'] != 1:  # 1表示人这一类
    #     continue

    categories = dict()
    for index in range(0, len(json_data["categories"]), 1):
        categories[json_data["categories"][index]["id"]] = json_data["categories"][index]["name"]
    # print(categories)

    bboxes = []
    category_id = []

    # 根据图片id，找对应的标注的image_id
    # print("图片id：", id)  # TODO 查找标注用id

    # count = 0
    for index in range(0, len(json_data["annotations"]), 1):
        if json_data["annotations"][index]["image_id"] == id:
            # count = count + 1
            # print("第", count, "个,标注image_id:", json_data["annotations"][index]["image_id"])

            # print("图片标注：", all_data["annotations"][index]["bbox"])
            # print("图片标注：", all_data["annotations"][index]["category_id"])

            bboxes.append(json_data["annotations"][index]["bbox"])
            category_id.append(json_data["annotations"][index]["category_id"])

    # 返回的图片信息比较全
    return img_name, img_height, img_width, bboxes, category_id, categories


def coco_visualize(img_dir, json_file, num_rate):
    # coco = COCO(json_file)    # TODO 可以不用open，使用pycocotools创建coco对象的方式来获取数据
    # json_data = coco.dataset

    with open(file=json_file, mode='r') as file:
        json_data = json.load(fp=file)

        # ====================================================================
        # ====================================================================
        # print("字典类型：", type(json_data))  # <class 'dict'>
        # print("字典长度：", len(json_data))  # 5
        #
        # print("字典的键：", json_data.keys())  # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
        # # # print("字典的值：", json_data.values())  #
        # # # print("字典的键值对：", json_data.items())  #

        # print("===========:", len(json_data["images"]))
        # print("===========:", len(json_data["annotations"]))
        # print("===========:", len(json_data["categories"]))

        # ====================================================================
        # ====================================================================
        # for image in json_data['images']:  # TODO 这个是以图片来遍历的
        #     print("图片名称:", image["file_name"], image["id"])
        #     for annotation in json_data['annotations']:
        #         if image["id"] == annotation["image_id"]:
        #             print("=================:", annotation["image_id"], annotation["category_id"])
        #             # print(annotation["id"])  # 总共标注
        #
        # # show all classes in coco
        # classes_names = []
        # classes_ids = []
        # classes = dict()
        # for cls in json_data['categories']:
        #     classes_ids.append(cls['id'])
        #     classes_names.append(cls['name'])
        #     classes[cls['id']] = cls['name']  # 组成字典
        #
        # print(classes)  # {1: 'hu', 2: 'shu'}
        # print(classes_names)  # ['hu', 'shu']
        # print(classes_ids)  # [1, 2]

        # ====================================================================
        # ====================================================================

        for index in range(0, int(num_rate * len(json_data['images'])), 1):  # TODO 根据index来遍历图片

            img_name, img_height, img_width, bboxes, category_id, categories = read_coco_json(json_data, index)

            # TODO 找到图片路径
            image_path = os.path.join(img_dir, img_name)  #
            img_mat = cv2.imread(filename=image_path)

            # TODO 把标注画在图上
            for index2 in range(0, len(bboxes), 1):
                xmin, ymin, width, height = bboxes[index2]  # 本来就是float数据

                point1 = (int(xmin), int(ymin))  # TODO 重点是找到这四个点
                point2 = (int(xmin + width), int(ymin))
                point3 = (int(xmin + width), int(ymin + height))
                point4 = (int(xmin), int(ymin + height))

                # 注意这里坐标必须为整数
                # TODO bboxes和标注没有建立对应关系，拿不到，以后在优化吧
                # cv2.line(img=img_mat, pt1=point1, pt2=point3, color=(0, 0, 255), thickness=2)
                # cv2.rectangle(img=img_mat, pt1=point1, pt2=point3, color=(0, 0, 255), thickness=2)
                cv2.putText(img=img_mat, text=categories[category_id[index2]], org=point3, fontFace=cv2.FONT_ITALIC,
                            fontScale=2, color=(0, 255, 0), thickness=5)

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

            # TODO 4.可视化
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


headstr = """\
<annotation>
    <folder>VOC2012</folder>
    <filename>%s</filename>
    <path>%s</path>
    <source>
        <database>My Database</database>
        <annotation>VOC2012</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def coco2voc(img_dir, json_file, num_rate, train_or_val, save_voc_dir):
    # coco = COCO(json_file)    # TODO 可以不用open，使用pycocotools创建coco对象的方式来获取数据
    # json_data = coco.dataset

    makedirs(save_voc_dir + "JPEGImages/")
    makedirs(save_voc_dir + "Annotations/")
    makedirs(save_voc_dir + "ImageSets/")

    with open(file=json_file, mode='r') as file:
        json_data = json.load(fp=file)

        for index in range(0, int(num_rate * len(json_data['images'])), 1):  # TODO 根据index来遍历图片

            img_name, img_height, img_width, bboxes, category_id, categories = read_coco_json(json_data, index)

            # classes = dict()
            # for item in json_data["categories"]:
            #     classes[item["id"]] = item["name"]

            makedirs(save_voc_dir + "ImageSets/" + "Main/")
            with open(file=save_voc_dir + "ImageSets/" + "Main/" + "{}.txt".format(train_or_val), mode="a") as file:
                file.write(img_name[:img_name.rfind(".")])
                file.write("\n")

            # 图片
            img_path = img_dir + img_name
            shutil.copy(src=img_path, dst=save_voc_dir + "JPEGImages/")  # copy会调用copyfile,但是copy的dst可以使目录

            # xml文件os.path.abspath(img_path)
            anno_path = save_voc_dir + "Annotations/" + img_name[:img_name.rfind(".")] + '.xml'

            img_mat = cv2.imread(filename=img_path)
            # print(img_name, img_width, img_height, img_mat.shape[2])
            # print(img_name, img.shape[1], img_mat.shape[0], img_mat.shape[2])
            assert img_width == img_mat.shape[1]
            assert img_height == img_mat.shape[0]
            ########## TODO 返回图片全路径，这里可以控制xml文件里面的path
            head = headstr % (img_name, os.path.abspath(img_path), img_mat.shape[1], img_mat.shape[0], img_mat.shape[2])
            objs = []
            for index2 in range(0, len(bboxes), 1):
                xmin, ymin, width, height = bboxes[index2]  # 本来就是float数据
                xmin = float(xmin)  ####################### TODO 数据处理
                ymin = float(ymin)  ####################### TODO 数据处理
                xmax = float(xmin + width)  ############### TODO 数据处理
                ymax = float(ymin + height)  ############## TODO 数据处理

                obj = [categories[category_id[index2]], xmin, ymin, xmax, ymax]
                objs.append(obj)
            tail = tailstr

            f = open(file=anno_path, mode="w")
            f.write(head)
            for index3 in range(0, len(objs), 1):
                f.write(objstr % (objs[index3][0], objs[index3][1], objs[index3][2], objs[index3][3], objs[index3][4]))
            f.write(tail)
