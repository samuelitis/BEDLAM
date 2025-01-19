import os
import sys
import argparse
from loguru import logger
from glob import glob
from train.core.testerx import Tester

import platform

# 운영 체제를 확인하고 환경 변수를 설정
if platform.system() == 'Linux':  # Linux에서는 EGL 사용 가능
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
elif platform.system() == 'Windows':  # Windows에서는 EGL 설정하지 않음
    print("EGL is not supported on Windows by default. Using the default platform.")
else:
    print(f"Operating system '{platform.system()}' is not explicitly supported for EGL. Using the default platform.")
sys.path.append('')

cfg='configs/demo_bedlam_cliff_x.yaml'
ckpt='data/ckpt/bedlam_cliff_x.ckpt'
image_folder='demo_images'
output_folder='results'
tracker_batch_size=1
detector='yolo'
yolo_img_size=416
dataframe_path=None
data_split='test'

tester = Tester(cfg,
                ckpt,
                image_folder,
                output_folder,
                tracker_batch_size,
                detector,
                yolo_img_size,
                dataframe_path,
                data_split)

if __name__ == '__main__':

    os.makedirs(output_folder, exist_ok=True)

    logger.add(
        os.path.join(output_folder, 'demo.log'),
        level='INFO',
        colorize=False,
    )

    all_image_folder = [image_folder]
    detections = tester.run_detector(all_image_folder)
    tester.run_on_image_folder(all_image_folder, detections, output_folder)

    del tester.model

    logger.info('================= END =================')