import numpy as np
from ..utils.box_utils import boxes3d_lidar_to_kitti_camera

def get_objects_from_label(label_file, calib):
    boxes = np.load(label_file)
    objects = [Object3d(bbox) for bbox in boxes3d_lidar_to_kitti_camera(boxes, calib)]
    return objects

class Object3d(object):
    def __init__(self, bbox):
        self.cls_type = "Object"
        self.cls_id = 1
        self.h = bbox[4]
        self.w = bbox[5]
        self.l = bbox[3]
        self.loc = bbox[:3]
        self.ry = bbox[6]

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str
