#!/bin/bash
# Устанавливаем ffmpeg через apt
apt-get update
apt-get install -y ffmpeg

# Устанавливаем Python-зависимости
pip install -r requirements.txt