import torch
import torch.nn as nn

from smplx import SMPLXLayer as SMPLX_
from smplx.utils import SMPLXOutput
from ...core import config


class SMPLX(SMPLX_):
    def __init__(self, *args, **kwargs):
        super(SMPLX, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        smplx_output = super(SMPLX, self).forward(*args, **kwargs)
        output = SMPLXOutput(vertices=smplx_output.vertices,
                             global_orient=smplx_output.global_orient,
                             body_pose=smplx_output.body_pose,
                             joints=smplx_output.joints,
                             betas=smplx_output.betas,
                             full_pose=smplx_output.full_pose)
        return output


class SMPLXHeadCamFull(nn.Module):
    def __init__(self, focal_length=5000., img_res=224):
        super(SMPLXHeadCamFull, self).__init__()    
        self.smplx = SMPLX(config.SMPLX_MODEL_DIR, flat_hand_mean=True, num_betas=11)
        self.add_module('smplx', self.smplx)
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, body_pose, lhand_pose, rhand_pose, shape, cam, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h, normalize_joints2d=False, T_pose=False):
        print(f'body_pose: type={type(body_pose)}, shape={body_pose.shape}, \n'
              f'lhand_pose: type={type(lhand_pose)}, shape={lhand_pose.shape}, \n'
              f'rhand_pose: type={type(rhand_pose)}, shape={rhand_pose.shape}, \n'
              f'shape: type={type(shape)}, shape={shape.shape}, \n'
              f'cam: type={type(cam)}, shape={cam.shape}, \n'
              f'cam_intrinsics: type={type(cam_intrinsics)}, shape={cam_intrinsics.shape}, \n'
              f'bbox_scale: type={type(bbox_scale)}, shape={bbox_scale.shape}, \n'
              f'bbox_center: type={type(bbox_center)}, shape={bbox_center.shape}, \n'
              f'img_w: type={type(img_w)}, value={img_w}, \n'
              f'img_h: type={type(img_h)}, value={img_h}, \n'
              f'normalize_joints2d: type={type(normalize_joints2d)}, value={normalize_joints2d}\n')

        # body_pose[0, 0, 1] = torch.tensor([
        #                         [1.0, 0.0, 0.0],
        #                         [0.0, 0.0, -1.0],
        #                         [0.0, 1.0, 0.0]
        #                     ], device='cuda:0')

        # lhand_pose[:, :, 0, :] = 2.0  # 첫 번째 벡터를 특정값으로 설정
        # rhand_pose[:, :, 0, :] = 3.0  # 첫 번째 벡터를 특정값으로 설정
        # Extract rotation matrix from body_pose
        for i in range(22):
            rotation_matrix = body_pose[0, i, :, :]
            euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

            # T_pose가 False일 때 원래 있는 값으로 회전
            if T_pose:
                modified_yaw = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
                modified_pitch = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
                if i == 0:
                    modified_roll = torch.deg2rad(torch.tensor(200.0, device=euler_angles.device))
                else:
                    modified_roll = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
            else:
                modified_yaw = euler_angles[0]  # 원래 있는 값 사용
                if i == 0:
                    modified_pitch = torch.deg2rad(torch.tensor(20.0, device=euler_angles.device))
                else:
                    modified_pitch = euler_angles[1]  # 원래 있는 값 사용
                modified_roll = euler_angles[2]  # 원래 있는 값 사용
                
                
            modified_rotation_matrix = euler_to_rotation_matrix(modified_yaw, modified_pitch, modified_roll)
            body_pose[0, i, :, :] = modified_rotation_matrix

        for i in range(15):
            rotation_matrix = lhand_pose[0, i, :, :]
            euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

            if T_pose:
                modified_yaw = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
                modified_pitch = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
                modified_roll = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
            else:
                modified_yaw = euler_angles[0]
                modified_pitch = euler_angles[1]
                modified_roll = euler_angles[2]

            modified_rotation_matrix = euler_to_rotation_matrix(modified_yaw, modified_pitch, modified_roll)
            lhand_pose[0, i, :, :] = modified_rotation_matrix
        for i in range(15):
            rotation_matrix = rhand_pose[0, i, :, :]
            euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

            if T_pose:
                modified_yaw = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
                modified_pitch = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
                modified_roll = torch.deg2rad(torch.tensor(0.0, device=euler_angles.device))
            else:
                modified_yaw = euler_angles[0]
                modified_pitch = euler_angles[1]
                modified_roll = euler_angles[2]

            modified_rotation_matrix = euler_to_rotation_matrix(modified_yaw, modified_pitch, modified_roll)
            rhand_pose[0, i, :, :] = modified_rotation_matrix

        smpl_output = self.smplx(
            betas=shape,
            body_pose=body_pose[:, 1:].contiguous(),
            global_orient=body_pose[:, 0].unsqueeze(1).contiguous(),
            left_hand_pose=lhand_pose.contiguous(),
            right_hand_pose=rhand_pose.contiguous(),
            pose2rot=False,
        )

        output = {
            'vertices': smpl_output.vertices,
            'joints3d': smpl_output.joints,
        }
        joints3d = output['joints3d']
        batch_size = joints3d.shape[0]
        device = joints3d.device

        cam_t = convert_pare_to_full_img_cam(
            pare_cam=cam,
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
            crop_res=self.img_res,
        )

        joints2d = perspective_projection(
            joints3d,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )
        if T_pose:
            print(cam_t)
            # cam_t = torch.tensor([[0., 0., 5]], device=device)
        
        output['joints2d'] = joints2d
        output['pred_cam_t'] = cam_t
        print(f' cam_t: {cam_t}')
        return output

def rotation_matrix_to_axis_angle(rotation_matrix):
    trace = torch.diagonal(rotation_matrix, dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos((trace - 1) / 2.0)  # Rotation angle

    # Prevent divide by zero for small angles
    theta = torch.where(theta < 1e-6, torch.tensor(1e-6, device=rotation_matrix.device), theta)

    # Extract rotation axis
    u_x = rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]
    u_y = rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]
    u_z = rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]

    axis = torch.stack([u_x, u_y, u_z], dim=-1)
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)  # Normalize

    return axis * theta.unsqueeze(-1)  # Axis-angle representation

def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = torch.sqrt(rotation_matrix[..., 0, 0]**2 + rotation_matrix[..., 1, 0]**2)
    singular = sy < 1e-6

    roll = torch.atan2(rotation_matrix[..., 2, 1], rotation_matrix[..., 2, 2])
    pitch = torch.where(
        ~singular,
        torch.atan2(-rotation_matrix[..., 2, 0], sy),
        torch.atan2(-rotation_matrix[..., 2, 0], torch.tensor(1e-6, device=rotation_matrix.device))
    )
    yaw = torch.atan2(rotation_matrix[..., 1, 0], rotation_matrix[..., 0, 0])

    return torch.stack([roll, pitch, yaw], dim=-1)
def euler_to_rotation_matrix(yaw, pitch, roll):
    R_z = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ])
    R_y = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ])
    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ])
    return R_z @ R_y @ R_x

def perspective_projection(points, rotation, translation, cam_intrinsics):
    K = cam_intrinsics
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:,:,-1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points.float())
    return projected_points[:, :, :-1]


def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t