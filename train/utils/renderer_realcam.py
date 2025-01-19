import os
import platform

from .vis_utils import draw_skeleton, visualize_joint_error, visualize_joint_uncertainty, visualize_heatmaps

# 운영 체제를 확인하고 환경 변수를 설정
if platform.system() == 'Linux':  # Linux에서는 EGL 사용 가능
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
elif platform.system() == 'Windows':  # Windows에서는 EGL 설정하지 않음
    print("EGL is not supported on Windows by default. Using the default platform.")
else:
    print(f"Operating system '{platform.system()}' is not explicitly supported for EGL. Using the default platform.")
# os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] \
#     if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

import torch
import trimesh
import pyrender
import numpy as np
from torchvision.utils import make_grid


class RealCamRenderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length, camera_center=(256,256), img_res=(512,512), faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                       viewport_height=img_res[1],
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = camera_center
        self.faces = faces

    def __call__(self, vertices, image, camera_pose):

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        # camera_translation[0] *= -1.


        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)


        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera = pyrender.IntrinsicsCamera(fx=self.focal_length[0], fy=self.focal_length[1],
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img