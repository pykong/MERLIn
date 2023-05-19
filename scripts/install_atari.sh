#!/bin/sh

ROM_DIR="roms/"

mkdir -p $ROM_DIR
poetry run AutoROM -y --install-dir $ROM_DIR
