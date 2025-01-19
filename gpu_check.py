import torch

# GPU가 사용 가능한지 확인
gpu_available = torch.cuda.is_available()
print(f"GPU 사용 가능 여부: {gpu_available}")

# 사용 가능한 GPU의 수
gpu_count = torch.cuda.device_count()
print(f"사용 가능한 GPU의 수: {gpu_count}")

# 현재 사용 중인 장치의 이름
if gpu_available:
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    print(f"현재 사용 중인 GPU: {gpu_name}")

