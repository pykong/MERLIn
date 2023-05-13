#!/bin/sh

nvidia-smi
poetry run python -c "import tensorflow as tf; print(tf.__version__)"
poetry run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
