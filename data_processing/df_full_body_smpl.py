import os
import smplx
import cv2
import torch
import pickle
import tqdm
import csv
import argparse
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from renderer_pyrd import Renderer

IMG_FORMAT = '.png'
SCALE_FACTOR_BBOX = 1.2
MODEL_FOLDER = 'bedlam_data/body_models/SMPL_python_v.1.1.0/smpl/models'

#To get imgname, gender, betas, poses, trans, cam_ext, cam_int, pitch, roll, yaw, center, scale, gtkp
imgnames, genders, betas, poses_cam, poses_world, trans_cam, trans_world, cam_ext, cam_int, cam_pitch, cam_roll, cam_yaw, body_yaw, \
centers, scales, gtkps, joints3d, proj_verts, motion_infos, subs = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

def get_bbox_valid(joints, img_height, img_width, rescale):
    #Get bbox using keypoints
    valid_j = []
    joints = np.copy(joints)
    for j in joints:
        if j[0] > img_width or j[1] > img_height or j[0] < 0 or j[1] < 0:
            continue
        else:
            valid_j.append(j)

    if len(valid_j) < 1:
        return [-1, -1], -1, len(valid_j), [-1, -1, -1, -1]

    joints = np.array(valid_j)

    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]

    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

    scale *= rescale
    return center, scale, len(valid_j), bbox


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def toCamCoords(j3d, camPosWorld):
    # transform gt to camera coordinate frame
    j3d = j3d - camPosWorld
    return j3d


def unreal2cv2(points):
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1.0, -1.0, 1.0])
    return points


def smpl2opencv(j3d):
    # change sign of axis 1 and axis 2
    j3d = j3d * np.array([1.0, -1.0, -1.0])
    return j3d


def get_cam_int(fl, sens_w, sens_h, cx, cy):
    flx = focalLength_mm2px(fl, sens_w, cx)
    fly = focalLength_mm2px(fl, sens_h, cy)

    cam_mat = np.array([[flx, 0, cx],
                       [0, fly, cy],
                       [0, 0, 1]])
    return cam_mat


smpl_model_male = smplx.SMPL(MODEL_FOLDER,
                                gender='male',
                                ext='npz',
                                num_betas=10)

smpl_model_female = smplx.SMPL(MODEL_FOLDER,
                                gender='female',
                                ext='npz',
                                num_betas=10)

smpl_model_neutral = smplx.SMPL(MODEL_FOLDER,
                                gender='neutral',
                                ext='npz',
                                num_betas=11)

def get_smpl_vertices(poses, betas, trans, gender):

    if gender == 'male':
        model_out = smpl_model_male(betas=torch.tensor(betas).unsqueeze(0).float(),
                              global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                              body_pose=torch.tensor(poses[3:]).unsqueeze(0).float(),
                              transl=torch.tensor(trans).unsqueeze(0))
# from psbody.mesh import Mesh
    elif gender == 'female':
        model_out = smpl_model_female(betas=torch.tensor(betas).unsqueeze(0).float(),
                              global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                              body_pose=torch.tensor(poses[3:]).unsqueeze(0).float(),
                              transl=torch.tensor(trans).unsqueeze(0))
    elif gender == 'neutral':
        model_out = smpl_model_neutral(betas=torch.tensor(betas).unsqueeze(0).float(),
                              global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                              body_pose=torch.tensor(poses[3:]).unsqueeze(0).float(),
                              transl=torch.tensor(trans).unsqueeze(0))
    else:
        print('Please provide gender as male or female')
    return model_out.vertices[0], model_out.joints[0]


def project(points, cam_trans, cam_int):
    points = points + cam_trans
    cam_int = torch.tensor(cam_int).float()

    projected_points = points / points[:, -1].unsqueeze(-1)
    projected_points = torch.einsum('ij, kj->ki', cam_int, projected_points.float())

    return projected_points.detach().cpu().numpy()


def get_cam_trans(body_trans, cam_trans):
    cam_trans = np.array(cam_trans) / 100
    cam_trans = unreal2cv2(np.reshape(cam_trans, (1, 3)))

    body_trans = np.array(body_trans) / 100
    body_trans = unreal2cv2(np.reshape(body_trans, (1, 3)))

    trans = body_trans - cam_trans
    return trans


def get_cam_rotmat(body_yaw, pitch, yaw, roll):
    #Because bodies are rotation by 90
    body_rotmat, _ = cv2.Rodrigues(np.array([[0, ((body_yaw - 90) / 180) * np.pi, 0]], dtype=float))
    rotmat_yaw, _ = cv2.Rodrigues(np.array([[0, ((yaw) / 180) * np.pi, 0]], dtype=float))
    rotmat_pitch, _ = cv2.Rodrigues(np.array([pitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    rotmat_roll, _ = cv2.Rodrigues(np.array([0, 0, roll / 180 * np.pi]).reshape(3, 1))
    final_rotmat = np.matmul(rotmat_roll, np.matmul(rotmat_pitch, rotmat_yaw))
    return body_rotmat, final_rotmat


def visualize(image_path, verts, focal_length, smpl_faces):
    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w, c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                        faces=smpl_faces)
    front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
                                            bg_img_rgb=img[:, :, ::-1].copy())
    cv2.imwrite(image_path.split('/')[-4]+image_path.split('/')[-1], front_view[:, :, ::-1])


def visualize_2d(image_path, joints2d):
    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = img[:, :, ::-1]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i in range(len(joints2d)):
        ax.scatter(joints2d[i, 0], joints2d[i, 1], s=0.2)
    plt.savefig(image_path.split('/')[-1])


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = rotate(new_img, rot) # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    from skimage.transform import rotate, resize

    # resize image
    new_img = resize(new_img, res) # scipy.misc.imresize(new_img, res)
    return img, new_img


def visualize_crop(image_path, center, scale, verts,focal_length, smpl_faces):
    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w, c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                        faces=smpl_faces)

    front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
                                            bg_img_rgb=img[:, :, ::-1].copy())

    img, crop_img = crop(front_view[:, :, ::-1], center, scale, res=(224, 224))
    cv2.imwrite(image_path.split('/')[-1], crop_img)


def get_global_orient(pose, beta, transl, gender, body_yaw, cam_pitch, cam_yaw, cam_roll, cam_trans):
    # World coordinate transformation after assuming camera has 0 yaw and is at origin
    body_rotmat, _ = cv2.Rodrigues(np.array([[0, ((body_yaw - 90+cam_yaw) / 180) * np.pi, 0]], dtype=float))
    pitch_rotmat, _ = cv2.Rodrigues(np.array([cam_pitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    roll_rotmat, _ = cv2.Rodrigues(np.array([0., 0, cam_roll / 180 * np.pi, ]).reshape(3, 1))
    final_rotmat = np.matmul(roll_rotmat, (pitch_rotmat))
    
    transform_coordinate = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    transform_body_rotmat = np.matmul(body_rotmat, transform_coordinate)
    w_global_orient = cv2.Rodrigues(np.dot(transform_body_rotmat, cv2.Rodrigues(pose[:3])[0]))[0].T[0]

    #apply rotation transformation to translation
    verts_local, joints_local = get_smpl_vertices(pose, beta, torch.zeros(3), gender)
    j0 = joints_local[0].detach().cpu().numpy()
    rot_j0 = np.matmul(transform_body_rotmat, j0.T).T
    l_translation_ = np.matmul(transform_body_rotmat, transl.T).T
    l_translation = rot_j0 + l_translation_
    w_translation = l_translation - j0

    c_global_orient = cv2.Rodrigues(np.dot(final_rotmat, cv2.Rodrigues(w_global_orient)[0]))[0].T[0]
    c_translation = np.matmul(final_rotmat, l_translation.T).T - j0 

    return w_global_orient, c_global_orient, c_translation, w_translation, final_rotmat


def get_params(image_folder, fl, start_frame, gender_sub, smpl_param_orig, trans_body, body_yaw_, cam_x, cam_y, cam_z, fps, sub, cam_pitch_=0., cam_roll_=0., cam_yaw_=0.  ):

    all_images = sorted(glob(os.path.join(image_folder, '*'+IMG_FORMAT)))
    every_fifth =- 4

    for img_ind, image_path in (enumerate(all_images)):
        # Saving every 5th frame
        every_fifth += 4
        if fps == 6:
            if img_ind % 5 != 0:
                continue
            smpl_param_ind = img_ind+start_frame
            cam_ind = img_ind
        else:
            smpl_param_ind = img_ind+start_frame
            cam_ind = img_ind

        if smpl_param_ind > smpl_param_orig['poses'].shape[0]:
            break
        pose = smpl_param_orig['poses'][smpl_param_ind]
        transl = smpl_param_orig['trans'][smpl_param_ind] 
        beta = smpl_param_orig['betas']
        motion_info = smpl_param_orig['motion_info']

        gender = smpl_param_orig['gender']
        cam_pitch_ind = -cam_pitch_[cam_ind]
        cam_yaw_ind = -cam_yaw_[cam_ind]
        if rotate_flag:
            cam_roll_ind = -cam_roll_[cam_ind] + 90
        else:
            cam_roll_ind = -cam_roll_[cam_ind]

        CAM_INT = get_cam_int(fl[cam_ind], SENSOR_W, SENSOR_H, IMG_W/2., IMG_H/2.)

        body_rotmat, cam_rotmat_for_trans = get_cam_rotmat(body_yaw_, cam_pitch_ind, cam_yaw_ind, cam_roll_ind)
        cam_t = [cam_x[cam_ind], cam_y[cam_ind], cam_z[cam_ind]]
        cam_trans = get_cam_trans(trans_body, cam_t)
        cam_trans = np.matmul(cam_rotmat_for_trans, cam_trans.T).T 
        
        w_global_orient, c_global_orient, c_trans, w_trans, cam_rotmat = get_global_orient(pose, beta, transl, gender, body_yaw_, cam_pitch_ind, cam_yaw_ind, cam_roll_ind, cam_trans)
        cam_ext_ = np.zeros((4, 4))
        cam_ext_[:3, :3] = cam_rotmat
        cam_ext_trans = np.concatenate([cam_trans, np.array([[1]])],axis=1)          
        cam_ext_[:, 3] = cam_ext_trans

        pose_cam = pose.copy()
        pose_cam[:3] = c_global_orient

        pose_world = pose.copy()
        pose_world[:3] = w_global_orient

        vertices3d, joints3d = get_smpl_vertices(pose_cam, beta, c_trans, gender)
        joints2d = project(joints3d, torch.tensor(cam_trans), CAM_INT)
        
        center, scale, num_vis_joints, bbox = get_bbox_valid(joints2d[:22], rescale=SCALE_FACTOR_BBOX, img_width=IMG_W, img_height=IMG_H)
        if center[0] < 0 or center[1] < 0 or scale <= 0:
            continue

        #visualize_crop(image_path, center, scale, torch.tensor(verts_cam2) , CAM_INT[0][0], smpl_model_male.faces)
        if num_vis_joints < 8:
            continue
        verts_cam = vertices3d.detach().cpu().numpy() + cam_trans
        if (verts_cam[:,2]<0).any():
            continue
        # verts_cam2 = vertices3d.detach().cpu().numpy() + cam_trans
        # visualize(image_path, torch.tensor(verts_cam2), CAM_INT[0][0], smpl_model_male.faces)
        # visualize_2d(image_path, joints2d)
        imgnames.append(os.path.join(image_path.split('/')[-2], image_path.split('/')[-1]))
        genders.append(gender_sub)
        betas.append(beta)
        poses_cam.append(pose_cam)
        poses_world.append(pose_world)
        trans_cam.append(c_trans)
        trans_world.append(w_trans)
        cam_ext.append(cam_ext_)
        cam_int.append(CAM_INT)
        gtkps.append(joints2d)
        centers.append(center)
        scales.append(scale)
        motion_infos.append(motion_info)
        subs.append(sub)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default='bedlam_data/images')
    parser.add_argument('--output_folder', type=str, default='bedlam_data/processed_labels')
    parser.add_argument('--smpl_gt_folder', type=str, default='bedlam_data/smpl_gt/smpl_ground_truth')
    parser.add_argument('--fps', type=int, default=6, help='6/30 fps output. With 6fps then every 5th frame is stored')

    args = parser.parse_args()
    base_image_folder = args.img_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gt_smpl_folder = args.smpl_gt_folder
    fps = args.fps

    image_folders = csv.reader(open('bedlam_scene_names.csv', 'r')) # File to parse folders
    next(image_folders) # Skip header
    image_dict = {}
    npz_dict = {}
    rotate_flag = False # Some of the images are output in rotated format by unreal
    for row in image_folders:
        image_dict[row[1]] = os.path.join(base_image_folder, row[0],'png')
        npz_dict[row[1]] = os.path.join(output_folder, str(row[0])+'.npz')

    for k, v in tqdm.tqdm(image_dict.items()):
        if 'closeup' in k:
            rotate_flag = True
            SENSOR_W = 20.25
            SENSOR_H = 36
            IMG_W = 720
            IMG_H = 1280
        else:
            rotate_flag = False
            SENSOR_W = 36
            SENSOR_H = 20.25
            IMG_W = 1280
            IMG_H = 720

        image_folder_base = v
        base_folder = v.replace('/png','')
        outfile = npz_dict[k]
        csv_path = os.path.join(base_folder, 'be_seq.csv')
        csv_data = pd.read_csv(csv_path)
        csv_data = csv_data.to_dict('list')

        seq_name = ''
        cam_csv_base = os.path.join(base_folder, 'ground_truth/camera')

        #To get imgname, gender, betas, poses, trans, cam_ext, cam_int, pitch, roll, yaw, center, scale, gtkp
        imgnames, genders, betas, poses_cam, poses_world, trans_cam, trans_world, cam_ext, cam_int, cam_pitch, cam_roll, cam_yaw, body_yaw, \
            centers, scales, gtkps, joints3d, motion_infos, subs = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        seq_name = ''
        # Parse csv file generated by Unreal for training
        for idx, comment in enumerate(csv_data['Comment']):
            if 'sequence_name' in comment:
                #Get sequence name and corresponding camera details
                seq_name = comment.split(';')[0].split('=')[-1]
                cam_csv_data = pd.read_csv(os.path.join(cam_csv_base, seq_name+'_camera.csv'))
                cam_csv_data = cam_csv_data.to_dict('list')
                cam_x = cam_csv_data['x']
                cam_y = cam_csv_data['y']
                cam_z = cam_csv_data['z']
                cam_yaw_ = cam_csv_data['yaw']
                cam_pitch_ = cam_csv_data['pitch']
                cam_roll_ = cam_csv_data['roll']
                fl = cam_csv_data['focal_length']
                continue
            elif 'start_frame' in comment:
                # Get body details
                start_frame = int(comment.split(';')[0].split('=')[-1])
                body = csv_data['Body'][idx]
                person_id_ = body.split('_')
                person_id = '_'.join(person_id_[:-1])
                sequence_id = person_id_[-1]
                smpl_param_orig = np.load(os.path.join(gt_smpl_folder, person_id, sequence_id, 'motion_seq.npz'))
                gender_sub = smpl_param_orig['gender_sub'].item()
                image_folder = os.path.join(image_folder_base, seq_name)
                X = csv_data['X'][idx]
                Y = csv_data['Y'][idx]
                Z = csv_data['Z'][idx]
                trans_body = [X, Y, Z]
                body_yaw_ = csv_data['Yaw'][idx]
                get_params(image_folder, fl, start_frame, gender_sub, smpl_param_orig, trans_body, body_yaw_, cam_x, cam_y, cam_z, fps, person_id, cam_pitch_=cam_pitch_, cam_roll_=cam_roll_, cam_yaw_=cam_yaw_  )
                break
            else:
                continue

        # np.savez(
        #     outfile,
        #     imgname=imgnames,
        #     center=centers,
        #     scale=scales,
        #     pose_cam=poses_cam,
        #     pose_world=poses_world,
        #     shape=betas,
        #     trans_cam=trans_cam,
        #     trans_world=trans_world,
        #     gtkps=gtkps,
        #     cam_int=cam_int,
        #     cam_ext=cam_ext,
        #     gender=genders,
        #     motion_info=motion_infos,
        #     sub=subs,
        # )

