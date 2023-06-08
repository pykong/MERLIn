#!/bin/sh

nvidia-smi
nvcc --version
poetry run python -c 'import torch; print(torch.__version__)'
poetry run python -c 'import torch; print(f"CUDA is available: {torch.cuda.is_available()}")'
poetry run python -c 'import torch; [print(f"{i}:{torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'
