
import os
import cv2
import json
import tqdm
import torch
import numpy as np
import pytorch_lightning as pl

from loguru import logger
from collections import OrderedDict
from multi_person_tracker import MPT
from torchvision.transforms import Normalize

from . import constants
from .config import update_hparams
from ..models.hmr import HMR
from ..utils.image_utils import crop_ul_br
from ..utils.renderer_pyrd import Renderer
from ..models.head.smplx_head_cam_full import SMPLXHeadCamFull
from ..models.hand import Hand
from ..models.hmrx import HMRX

from kornia.geometry.transform.imgwarp import (
    get_perspective_transform, warp_affine
)

SCALE_FACTOR_HAND_BBOX=3.0
MIN_NUM_FRAMES = 0

class SMPLXTrainer(pl.LightningModule):
    def __init__(self, hparams, config_tune=None):
        super(SMPLXTrainer, self).__init__()
        self.hparams.update(hparams)

        self.body_model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )
        self.hand_model = Hand(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )
        self.fullbody_model = HMRX(backbone=self.hparams.MODEL.BACKBONE,
                                   img_res=self.hparams.DATASET.IMG_RES,
                                   pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
                                   hparams=self.hparams)
                    
def get_bbox_valid(joints, img_height, img_width, rescale):
    min_coords, _ = torch.min(joints, dim=1)
    xmin, ymin = min_coords[:, 0], min_coords[:, 1]
    max_coords, _ = torch.max(joints, dim=1)
    xmax, ymax = max_coords[:, 0], max_coords[:, 1]
    center = torch.stack([xmax + xmin, ymax + ymin], dim=-1) * 0.5
    width = (xmax - xmin)
    height = (ymax - ymin)
    # Convert the bounding box to a square box
    scale = torch.max(width, height).unsqueeze(-1)
    scale *= rescale
    return center, scale


def crop_tensor(image, center, bbox_size, crop_size, interpolation = 'bilinear', align_corners=False):

    dtype = image.dtype
    device = image.device
    batch_size = image.shape[0]
    # points: top-left, top-right, bottom-right, bottom-left    
    src_pts = torch.zeros([4,2], dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    src_pts[:, 0, :] = center - bbox_size*0.5  # / (self.crop_size - 1)
    src_pts[:, 1, 0] = center[:, 0] + bbox_size[:, 0] * 0.5
    src_pts[:, 1, 1] = center[:, 1] - bbox_size[:, 0] * 0.5
    src_pts[:, 2, :] = center + bbox_size * 0.5
    src_pts[:, 3, 0] = center[:, 0] - bbox_size[:, 0] * 0.5
    src_pts[:, 3, 1] = center[:, 1] + bbox_size[:, 0] * 0.5

    DST_PTS = torch.tensor([[
        [0, 0],
        [crop_size - 1, 0],
        [crop_size - 1, crop_size - 1],
        [0, crop_size - 1],
    ]], dtype=dtype, device=device).expand(batch_size, -1, -1)
    # estimate transformation between points
    dst_trans_src = get_perspective_transform(src_pts, DST_PTS)
    # simulate broadcasting
    # dst_trans_src = dst_trans_src.expand(batch_size, -1, -1)

    # warp images 
    cropped_image = warp_affine(
        image, dst_trans_src[:, :2, :], (crop_size, crop_size),
        mode=interpolation, align_corners=align_corners)

    tform = torch.transpose(dst_trans_src, 2, 1)
    return cropped_image, tform

class Tester:
    def __init__(self, cfg, ckpt, image_folder, output_folder, tracker_batch_size, detector, yolo_img_size, dataframe_path, data_split):
        self.cfg = cfg
        self.ckpt = ckpt
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.tracker_batch_size = tracker_batch_size
        self.detector = detector
        self.yolo_img_size = yolo_img_size
        self.dataframe_path = dataframe_path
        self.data_split = data_split
        self.model_cfg = update_hparams(cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        self._build_model()
        self._load_pretrained_model()
        self.model.eval()

        self.smpl_cam_head = SMPLXHeadCamFull(img_res=self.model_cfg.DATASET.IMG_RES).to(self.device)

        self.flip_vector = torch.ones((1, 9), dtype=torch.float32)
        self.flip_vector[:, [1, 2, 3, 6]] *= -1
        self.flip_vector = self.flip_vector.reshape(1, 3, 3).cuda()
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)


    def _build_model(self):
        # ========= Define SPEC model ========= #
        self.hparams = self.model_cfg
        self.model = SMPLXTrainer(hparams=self.hparams).to(self.device)

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        logger.info(f'Loading pretrained model from {self.ckpt}')
        ckpt = torch.load(self.ckpt)['state_dict']
        load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
        logger.info(f'Loaded pretrained weights from \"{self.ckpt}\"')

    def run_detector(self, all_image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.tracker_batch_size,
            display=False,
            detector_type=self.detector,
            output_format='dict',
            yolo_img_size=self.yolo_img_size,
        )
        bboxes = []
        for _, image_folder in enumerate(all_image_folder):
            bboxes.append(mot.detect(image_folder))

        return bboxes

    @torch.no_grad()
    def run_on_image_folder(self, all_image_folder, detections, output_folder):
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
            ]
            image_file_names = (sorted(image_file_names))

            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                dets = detections[fold_idx][img_idx]
                if len(dets) < 1:
                    continue

                if len(dets.shape)==1:
                    dets = np.expand_dims(dets, 0)
                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                rgb_img_full = np.transpose(img.astype('float32'),(2,0,1))/255.0

                orig_height, orig_width = img.shape[:2]

                inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
                                        self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

                batch_size = inp_images.shape[0]

                bbox_scale = []
                bbox_center = []
 
                for det_idx, det in enumerate(dets):
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img, ul, br = crop_ul_br(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
                    rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)


                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()

                body_pred = self.model.body_model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)

                lhand_joints = body_pred['joints2d'][:, 25:40]
                rhand_joints = body_pred['joints2d'][:, 40:55]

                center_r, scale_r = get_bbox_valid(rhand_joints, img_h, img_w, SCALE_FACTOR_HAND_BBOX)

                rgb_img_full = torch.tensor(rgb_img_full).cuda().float()
                rgb_img_full = rgb_img_full.unsqueeze(0).expand(inp_images.shape[0],-1,-1,-1)
                right_hand_crop, _ = crop_tensor(rgb_img_full, center_r, scale_r, 224)
                right_hand_crop = self.normalize_img(right_hand_crop)
                
                right_hand_pred = self.model.hand_model(right_hand_crop)
                
                #Flip left hand image before feedint to hand network
                center_l, scale_l = get_bbox_valid(lhand_joints, img_h, img_w, SCALE_FACTOR_HAND_BBOX)
                left_hand_crop, _ = crop_tensor(rgb_img_full, center_l, scale_l, 224)
                left_hand_crop = self.normalize_img(left_hand_crop)
                left_hand_crop = torch.flip(left_hand_crop, [3])
                
                left_hand_pred = self.model.hand_model(left_hand_crop)

                #Flip predicted right hand pose to left hand 
                left_hand_pred['pred_pose'] = left_hand_pred['pred_pose'] * self.flip_vector.unsqueeze(0)

                full_body_pred = self.model.fullbody_model(body_pred['body_feat'], 
                                                           left_hand_pred['hand_feat'], 
                                                           right_hand_pred['hand_feat'], 
                                                           body_pred['pred_pose'], 
                                                           body_pred['pred_shape'], 
                                                           body_pred['pred_cam'],
                                                           left_hand_pred['pred_pose'], 
                                                           right_hand_pred['pred_pose'],
                                                           bbox_center, 
                                                           bbox_scale, 
                                                           img_w, 
                                                           img_h
                                                           )
                
                cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
                cam_intrinsics[:, 0, 0]  = focal_length
                cam_intrinsics[:, 1, 1]  = focal_length
                cam_intrinsics[:, 0, 2] = img_w/2.
                cam_intrinsics[:, 1, 2] = img_h/2.

                output = self.smpl_cam_head(body_pose=full_body_pred['pred_pose'], 
                                            lhand_pose = left_hand_pred['pred_pose'][:,1:], 
                                            rhand_pose=right_hand_pred['pred_pose'][:,1:],
                                            shape=body_pred['pred_shape'], 
                                            cam=body_pred['pred_cam'], 
                                            cam_intrinsics=cam_intrinsics, 
                                            bbox_scale=bbox_scale,
                                            bbox_center=bbox_center, 
                                            img_w=img_w, 
                                            img_h=img_h, 
                                            normalize_joints2d=False)

                f_name = img_fname.split('\\')[-1].split('.')[-2] + '.json'
                output_file_path = os.path.join(output_folder, f_name)
                save_output_to_file(output, output_file_path)

                del inp_images

                # ========== render & write ==========
                img_h, img_w, _ = img.shape
                focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                vertices = output['vertices'].detach().cpu().numpy()
                translation = output['pred_cam_t'].detach().cpu().numpy()

                pred_vertices_array = vertices + np.expand_dims(translation, 1)
                renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                    faces=self.smpl_cam_head.smplx.faces,
                                    same_mesh_color=False)

                front_view = renderer.render_front_view(pred_vertices_array,
                                                        # bg_img_rgb=img.copy()
                                                        )
                # save rendering results
                basename = img_fname.split('\\')[-1]
                filename = basename
                front_view_path = os.path.join(output_folder, filename)
                logger.info(f'Writing output files to {output_folder}')
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])

                renderer.delete()
    def render(self, x, output_folder, bg=None):
        img_h, img_w, _ = 256, 256, 3
        focal_length = (img_w * img_w + img_h * img_h) ** 0.5

        vertices = x['vertices'].detach().cpu().numpy()
        translation = x['pred_cam_t'].detach().cpu().numpy()

        pred_vertices_array = vertices + np.expand_dims(translation, 1)
        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=self.smpl_cam_head.smplx.faces,
                            same_mesh_color=False)

        front_view = renderer.render_front_view(pred_vertices_array,)

        renderer.delete()
        return front_view

def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True, remove_lightning=False):
    if remove_lightning:
        logger.warning(f'Removing "model." keyword from state_dict keys..')
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]

        model.load_state_dict(new_state_dict, strict=strict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        logger.warning(f'size mismatch for \"{pk}\": copying a param with shape {state_dict[pk].shape} '
                                       f'from checkpoint, the shape in current model is {model_state_dict[pk].shape}')

                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat(
                                [state_dict[pk], state_dict[pk][:,-7:]], dim=-1
                            )
                            logger.warning(f'Updated \"{pk}\" param to {updated_pretrained_state_dict[pk].shape} ')
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')
    return model

def save_output_to_file(output, file_path):
    output_dict = {key: value.cpu().numpy().tolist() for key, value in output.items()}
    
    with open(file_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

def load_output_from_file(file_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    with open(file_path, 'r') as f:
        output_dict = json.load(f)
    
    output = {key: torch.tensor(value, device=device) for key, value in output_dict.items()}
    return output