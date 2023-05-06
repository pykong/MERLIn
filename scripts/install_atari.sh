#!/bin/sh

ROM_DIR="docs/"

mkdir -p $ROM_DIR
poetry run AutoROM -y --install-dir $ROM_DIR
