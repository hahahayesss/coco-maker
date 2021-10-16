import os
import cv2
import click
import numpy as np

from multiprocessing import Pool

BOUNDARIES = [
    {
        "id": 1,
        "name": "Bus",
        "supercategory": "vehicle",
        "start": [0, 1, 1],
        "end": [0, 1, 255]
    }, {
        "id": 2,
        "name": "Car",
        "supercategory": "vehicle",
        "start": [0, 2, 1],
        "end": [0, 2, 255]
    }, {
        "id": 3,
        "name": "Lorry",
        "supercategory": "vehicle",
        "start": [0, 3, 1],
        "end": [0, 3, 255]
    }, {
        "id": 4,
        "name": "Truck",
        "supercategory": "vehicle",
        "start": [0, 4, 1],
        "end": [0, 4, 255]
    }, {
        "id": 5,
        "name": "Van",
        "supercategory": "vehicle",
        "start": [0, 5, 1],
        "end": [0, 5, 255]
    }
]

dataset = [{
    "image": "dataset/images/0.png",
    "mask": {
        "image": "dataset/masks/0.png",
        "keys": [{
            "color": [0, 1, 255],
            "segment": []
        }]
    }
}, {
    "image": "dataset/images/1.png",
    "mask": {
        "image": "dataset/masks/1.png",
        "keys": [{
            "color": [0, 1, 254],
            "segment": []
        }, {
            "color": [0, 1, 255],
            "segment": []
        }, {
            "color": [0, 1, 253],
            "segment": []
        }, {
            "color": [0, 1, 252],
            "segment": []
        }]
    }
}]


def __create_id(color: np.ndarray) -> str:
    return str(color[0]) + ";" + str(color[1]) + ";" + str(color[2]) #


def __sorted(ds: list) -> list:
    _names = [_x["image"] for _x in ds]
    _names = sorted(_names)

    temp_ds = []
    for _name in _names:
        temp_ds.append(
            next(_x for _x in ds if _x["image"] == _name))
    return temp_ds


def _get_folder_structure(dataset) -> (str, str):
    assert os.path.exists(dataset), "Dataset folder not exists"
    images_folder = os.path.join(dataset, "images")
    assert os.path.isdir(images_folder), "Images folder not found"
    masks_folder = os.path.join(dataset, "masks")
    assert os.path.isdir(masks_folder), "Masks folder not found"
    return images_folder, masks_folder


def _find_segments(mask: np.ndarray) -> list:
    mask = mask.reshape((mask.shape[0] * mask.shape[1], 3))
    return np.unique(mask, axis=0)[1:]


def _mask_to_segment(image: np.ndarray, color_code: np.ndarray) -> np.ndarray:
    mask = image.reshape((-1, 3))

    cuts = np.asarray(list(map(lambda _pixel: np.array_equal(_pixel, color_code), mask)))
    cuts = np.flatnonzero(np.diff(cuts))
    cuts = np.hstack([cuts + 1, mask.shape[0]])

    result = [cuts[0]]
    for x in range(1, cuts.shape[0]):
        result.append(cuts[x] - cuts[x - 1])
    return np.asarray(result)[:-1]


# =====================================================================================================================
def create_ds(pair: list) -> dict:
    mask_image = cv2.imread(pair[1])[:, :, ::-1]
    segments_colors = _find_segments(mask_image)

    annotations = []
    for _y, color in enumerate(segments_colors):
        annotations.append({
            "id": __create_id(color),
            "color": color.tolist(),
            "category_id": color[1],
            "track_id": -1,
            "segmentation": _mask_to_segment(mask_image, color).tolist()
        })

    return {
        "image": pair[0],
        "mask": pair[1],
        "annotations": annotations
    }


def arrange_ids(dataset: list, io: int, ao: int, to: int) -> list:
    _tracking_ids = {}
    for _x, image in enumerate(dataset):
        dataset[_x]["id"] = io
        io += 1

        for _y, annotation in enumerate(image["annotations"]):
            if annotation["id"] in _tracking_ids:
                dataset[_x]["annotations"][_y]["track_id"] = _tracking_ids[annotation["id"]]
            else:
                _tracking_ids[annotation["id"]] = to
                dataset[_x]["annotations"][_y]["track_id"] = to
                to += 1

            dataset[_x]["annotations"][_y]["id"] = ao
            ao += 1

    print("[INFO]: Next image id will be : ", io)
    print("[INFO]: Next annotation id will be : ", ao)
    print("[INFO]: Next tracking id will be : ", to)
    return dataset


def _for_check(image: np.ndarray, segment: np.ndarray) -> np.ndarray:
    FILL_WITH = [255, 255, 255]
    mask = None
    for x in range(0, segment.shape[0], 2):
        temp = np.full(shape=(segment[x], 3), fill_value=[0, 0, 0])
        if mask is None:
            mask = temp
        else:
            mask = np.append(mask, temp, axis=0)

        temp = np.full(shape=(segment[x + 1], 3), fill_value=FILL_WITH)
        mask = np.append(mask, temp, axis=0)

    missing = (1920 * 1080) - mask.shape[0]
    temp = np.full(shape=(missing, 3), fill_value=[0, 0, 0])
    mask = np.append(mask, temp, axis=0)

    mask = mask.flatten()
    image = image.flatten()

    image = np.add(image, mask)
    image = np.asarray(
        list(map(lambda _x: 255 if _x > 255 else _x, image))
    )
    image = image.reshape((1080, 1920, 3))
    return image


# ====
@click.command()
# ====
@click.option("--dataset", default="dataset", help="Dataset location")
@click.option("--process", default=8, help="Process count")
@click.option("--io", default=1, help="Image id offset")
@click.option("--ao", default=1, help="Annotation id offset")
@click.option("--to", default=1, help="Tracking id offset")
def main(dataset, process, io, ao, to):
    images_folder, masks_folder = _get_folder_structure(dataset)
    images_files = os.listdir(images_folder)
    masks_files = os.listdir(masks_folder)
    assert len(images_files) >= len(masks_files), "Dataset not contains enough masks"

    data_pairs = []
    names = [x.split(".")[0] for x in images_files]
    for name in names:
        data_pairs.append([
            os.path.join(images_folder,
                         next(x for x in images_files if x.split(".")[0] == name)),
            os.path.join(masks_folder,
                         next(x for x in masks_files if x.split(".")[0] == name))
        ])

    with Pool(process) as p:
        ds = p.map(create_ds, data_pairs)
    ds = __sorted(ds)
    ds = arrange_ids(ds, io, ao, to)
    print(ds)


def start():
    """
    image = cv2.imread(dataset[1]["mask"]["image"])[:, :, ::-1]
    image = image.reshape((1080 * 1920, 3))
    print(image.shape)
    temp = np.unique(image, axis=0)
    print(temp)
    """

    image = cv2.imread("dataset/images/0.png")[:, :, ::-1]
    segment = [822868, 4, 1914, 9, 1908, 15, 1905, 17, 1902, 20, 1900, 21, 1898, 23, 1897, 23, 1897, 24, 1896, 24, 1897,
               23, 1896, 24, 1896, 24, 1895, 25, 1895, 24, 1896, 24, 1895, 25, 1895, 24, 1895, 24, 1896, 24, 1896, 24,
               1895, 24, 1896, 24, 1895, 25, 1893, 26, 1895, 25, 1895, 24, 1896, 24, 1895, 24, 1896, 24, 1895, 25, 1895,
               24, 1895, 25, 1894, 28, 1892, 27, 1892, 26, 1894, 25, 1894, 26, 1894, 25, 1895, 25, 1895, 25, 1895, 25,
               1895, 24, 1897, 23, 1897, 23, 1898, 21, 1901, 19, 1902, 17, 1905, 15, 1906, 12, 1914, 3]
    segment = np.asarray(segment)
    image = _for_check(image, segment)[:, :, ::-1]
    image = cv2.blur(image, (3, 3))
    cv2.imwrite("temp.png", image)

    """
        for _x, data in enumerate(dataset):
        _mask_image = cv2.imread(data["mask"]["image"])[:, :, ::-1]
        for _y, key in enumerate(data["mask"]["keys"]):
            dataset[_x]["mask"]["keys"][_y]["segment"] = mask_to_segment(
                _mask_image, np.asarray(key["color"])
            )
    """


if __name__ == '__main__':
    main()
