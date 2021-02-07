import cv2
import json
import sys
import os
import zlib
import base64
import numpy as np


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def get_ds_names(folder_name):
    try:
        list = os.listdir(os.path.join("data", folder_name))
        paths = []
        for name in list:
            cur_path = os.path.join("data", folder_name, name)
            if os.path.isdir(cur_path):
                paths.append(cur_path)
        return paths
    except:
        print("Folder not exist")


def get_masks_from_json(json_path):
    ret_masks = []
    filename = json_path
    with open(json_path, "r") as read_file:
        data = json.load(read_file)
        h = data["size"]["height"]
        w = data["size"]["width"]
        objects = data["objects"]
        objects_dict = {}
        for object in objects:
            key = object["key"]
            label = object["classTitle"]
            objects_dict[key] = label

        frames = data["frames"]
        for frame in frames:
            number = frame["index"]
            figures = frame["figures"]
            for figure in figures:
                label_key = figure["objectKey"]
                label = "unknown"
                if label_key in objects_dict:
                    label = objects_dict[label_key]
                left, top = figure["geometry"]["bitmap"]["origin"]
                base64_data = figure["geometry"]["bitmap"]["data"]
                mask = base64_2_mask(base64_data)
                mask_full = np.zeros((h, w), dtype=bool)
                mask_full[
                    top : (top + mask.shape[0]), left : (left + mask.shape[1])
                ] = mask
                ret_masks.append(
                    {
                        "filename": filename,
                        "width": w,
                        "heigh": h,
                        "number": number,
                        "label": label,
                        "mask": mask_full,
                        "left": left,
                        "top": top,
                    }
                )
    return ret_masks


video = sys.argv[1]
video_file_path = "data/test_rsl/ds0/video/" + video
json_file_path = "data/test_rsl/ds0/ann/" + video + ".json"
cap = cv2.VideoCapture(video_file_path)
masks = get_masks_from_json(json_file_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
result = cv2.VideoWriter(
    "data/test_rsl/ds0/result/" + video + ".avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1080, 240),
)

if cap.isOpened() == False:
    print("Error open video!")

frame_mask = np.array([])
number_prev = 0
count_frame_mask = 0
number = 0
for i in range(len(masks)):
    if number_prev != masks[i]["number"]:
        frame_mask = np.append(frame_mask, masks[i])
        frame_mask[number]["mask"] = frame_mask[number]["mask"].astype(np.uint8) * 255
        frame_mask[number]["mask"] = cv2.cvtColor(
            frame_mask[number]["mask"], cv2.COLOR_GRAY2RGB
        )
        count_frame_mask += 1
        number_prev = masks[i]["number"]
        number += 1
    else:
        t = masks[i]["mask"].astype(np.uint8) * 255
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)
        cv2.addWeighted(
            t,
            0.6,
            frame_mask[number - 1]["mask"],
            1.0,
            0,
            frame_mask[number - 1]["mask"],
        )

i = 0
j = 0
black_frame = np.array([np.zeros([240 * 360 * 3])])
black_frame = black_frame.astype(np.uint8)
black_frame = black_frame.reshape(240, 360, 3)

while cap.isOpened():
    ret, img = cap.read()

    if ret == True:
        # img = cv2.rotate(img, cv2.ROTATE_180)

        if i == frame_mask[j]["number"]:
            frame_vid_180 = img.copy()
            mask = frame_mask[j]["mask"]
            cv2.addWeighted(mask, 0.4, frame_vid_180, 1.0, 0, frame_vid_180)
            mask = cv2.resize(mask, (360, 240))
            frame_vid_180 = cv2.resize(frame_vid_180, (360, 240))
            if j != count_frame_mask - 1:
                j += 1
        else:
            mask = black_frame
            frame_vid_180 = cv2.resize(img, (360, 240))

        img = cv2.resize(img, (360, 240))
        i += 1

        conc_frames = np.concatenate((img, frame_vid_180), axis=1)
        conc_frames = np.concatenate((conc_frames, mask), axis=1)
        result.write(conc_frames)
        # cv2.imshow('Video', conc_frames) #shape 240, 1080, 3

    if cv2.waitKey(fps) & ret == False:
        break

cap.release()
result.release()
cv2.destroyAllWindows()

cv2.waitKey(0)
