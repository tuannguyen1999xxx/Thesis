from torchreid.utils import FeatureExtractor
import os
import torch


def feature_extract_one(path):
    list_path_image = []
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        list_path_image.append(image_path)
    extractor = FeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path='osnet_ain_x1_0_imagenet.pth',
        device='cpu'
    )
    features = extractor(list_path_image)
    return features


def feature_extract_all(path):
    list_path_image = []
    list_id = []
    for id in os.listdir(path):
        path_temp = os.path.join(path, id)
        for image_name in os.listdir(path_temp):
            image_path = os.path.join(path_temp, image_name)
            list_path_image.append(image_path)
            list_id.append(id)

    extractor = FeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path='osnet_ain_x1_0_imagenet.pth',
        device='cpu'
    )
    features = extractor(list_path_image)
    return features, list_id


def add_id(list_dist, list_ids):
    result = []
    for i in range(len(list_dist)):
        temp = list(list_dist[i])
        temp.append(list_ids[i])
        result.append(temp)
    return result


def most_frequent(List):
    return max(set(List), key=List.count)


def list_common_id(list_dist):
    list_commonid = []
    for i in list_dist:
        list_commonid.append(i[1])
    return list_commonid


def reid(feature, database_vector, list_ids):
    feature = feature.reshape(1, -1)
    list_dist = []

    for i in database_vector:
        feat = i.reshape(1, -1)
        list_dist.append(torch.cdist(feature, feat))
    list_dist_and_ids = add_id(list_dist, list_ids)
    list_dist_and_ids.sort(key=lambda x: x[0])

    list_shortest_dist = list_dist_and_ids[0:5]
    list_commonid = list_common_id(list_shortest_dist)

    re_id = most_frequent(list_commonid)
    return re_id
