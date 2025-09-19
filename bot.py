from dotenv import load_dotenv
import os
import asyncio
import logging
import subprocess
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from PIL import Image
import cv2
import imageio

load_dotenv()  
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("❌ Токен бота не найден. Проверь файл .env")

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Папка для сохранения результатов
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Пути к изображениям помидора
TOMATO_WHOLE_PATH = "tomato_whole.png"    # Целый помидор
TOMATO_EXPLODED_PATH = "tomato_exploded.png"  # Взорванный помидор

# Пути к звукам
WHOOSH_SOUND_PATH = "whoosh.mp3"    # Звук полёта
BOOM_SOUND_PATH = "boom.mp3"        # Звук взрыва

logging.basicConfig(level=logging.INFO)


@dp.message(Command("start"))
async def send_welcome(message: Message):
    await message.reply("Привет! Отправь мне фото!")


@dp.message(F.photo)
async def handle_photo(message: Message):
    if not message.photo:
        return

    await message.reply("⏳ Изображение в работе...")

    # Получаем файл фото
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_path = file_info.file_path

    # Скачиваем фото
    downloaded_file = await bot.download_file(file_path)

    # Сохраняем фото во временный файл
    input_photo_path = os.path.join(OUTPUT_DIR, f"{message.from_user.id}_input.jpg")
    with open(input_photo_path, 'wb') as f:
        f.write(downloaded_file.getvalue())

    # Путь для результата
    output_video_path = os.path.join(OUTPUT_DIR, f"{message.from_user.id}_output.mp4")

    try:
        # Создаём видео с летящим помидором и звуками
        create_flying_tomato_video(
            background_path=input_photo_path,
            tomato_whole_path=TOMATO_WHOLE_PATH,
            tomato_exploded_path=TOMATO_EXPLODED_PATH,
            whoosh_sound_path=WHOOSH_SOUND_PATH,
            boom_sound_path=BOOM_SOUND_PATH,
            output_path=output_video_path
        )

        # Отправляем результат как видео
        video = FSInputFile(output_video_path)
        await bot.send_video(
            chat_id=message.chat.id,
            video=video,
            caption="Лови результат!"
        )

        # Удаляем временные файлы
        os.remove(input_photo_path)
        os.remove(output_video_path)

    except Exception as e:
        await message.reply(f"❌ Ошибка: {str(e)}")
        logging.error(f"Ошибка при обработке: {e}")


def create_flying_tomato_video(background_path, tomato_whole_path, tomato_exploded_path, whoosh_sound_path, boom_sound_path, output_path):
    # Загружаем фон
    background = Image.open(background_path).convert("RGBA")
    bg_width, bg_height = background.size

    # Округляем размеры до чётных чисел (требование libx264)
    bg_width = bg_width - (bg_width % 2)
    bg_height = bg_height - (bg_height % 2)
    background = background.resize((bg_width, bg_height), Image.LANCZOS)

    # Конвертируем в OpenCV для поиска лица
    background_cv = cv2.cvtColor(np.array(background), cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(background_cv, cv2.COLOR_BGR2GRAY)

    # Загружаем каскад для лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Определяем цель: центр первого лица
    if len(faces) > 0:
        x, y, w, h = faces[0]
        target_x = x + w // 2
        target_y = y + h // 2
        face_size = max(w, h)
        print(f"🎯 Найдено лицо! Цель: ({target_x}, {target_y}), размер: {face_size}")
    else:
        target_x = bg_width // 2
        target_y = bg_height // 2
        face_size = min(bg_width, bg_height) // 3
        print("😐 Лицо не найдено. Летим в центр.")

    # Загружаем изображения помидора
    tomato_whole = Image.open(tomato_whole_path).convert("RGBA")
    tomato_exploded = Image.open(tomato_exploded_path).convert("RGBA")

    # Настройки анимации
    fps = 24
    flight_duration = 1.5  # Длительность полёта
    freeze_duration = 2.0   # Сколько секунд держать "взрыв" на лице
    total_duration = flight_duration + freeze_duration
    total_output_frames = int(total_duration * fps)

    frames_for_video = []

    for i in range(total_output_frames):
        t = i / total_output_frames  # 0.0 → 1.0 по всей длительности

        # Определяем, в какой фазе мы: полёт или "заморозка"
        if t <= (flight_duration / total_duration):
            # Фаза полёта
            progress = t / (flight_duration / total_duration)  # 0.0 → 1.0 внутри фазы полёта
            tomato_frame = tomato_whole.copy()
        else:
            # Фаза "заморозки" — показываем взорванный помидор
            progress = 1.0
            tomato_frame = tomato_exploded.copy()

        # Масштабируем помидор относительно размера лица
        target_tomato_size = int(face_size * 1.2)  # Множитель 1.2 — можно настроить
        original_max = max(tomato_frame.size)
        scale_factor = target_tomato_size / original_max
        new_width = int(tomato_frame.width * scale_factor)
        new_height = int(tomato_frame.height * scale_factor)
        tomato_frame = tomato_frame.resize((new_width, new_height), Image.LANCZOS)

        # Создаем копию фона
        frame_background = background.copy()

        # Анимация: помидор летит снизу → в цель
        start_x = bg_width // 2 - new_width // 2
        start_y = bg_height + new_height  # начинаем СНИЗУ экрана
        x = int(start_x + (target_x - new_width // 2 - start_x) * progress)
        y = int(start_y + (target_y - new_height // 2 - start_y) * progress)

        # Накладываем помидор
        frame_background.paste(tomato_frame, (x, y), tomato_frame)
        frame_background = frame_background.convert("RGB")
        frames_for_video.append(np.array(frame_background))

    # Сохраняем видео без звука
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")
    imageio.mimwrite(temp_video_path, frames_for_video, fps=fps, codec='libx264', macro_block_size=None)

    # === ДОБАВЛЯЕМ ЗВУКИ ЧЕРЕЗ FFMPEG ===
    # Проверяем наличие звуковых файлов
    if not os.path.exists(whoosh_sound_path) or not os.path.exists(boom_sound_path):
        print("⚠️ Звуковые файлы не найдены, возвращаем видео без звука")
        os.rename(temp_video_path, output_path)
        return

    # Создаём команду ffmpeg для наложения звуков
    cmd = [
        'ffmpeg',
        '-y',  # перезаписать без подтверждения
        '-i', temp_video_path,
        '-i', whoosh_sound_path,
        '-i', boom_sound_path,
        '-filter_complex',
        f'[1:a]adelay=0|0[a1];'  # whoosh начинается сразу
        f'[2:a]adelay={int(flight_duration * 1000)}|{int(flight_duration * 1000)}[a2];'  # boom начинается через flight_duration секунд
        f'[a1][a2]amix=inputs=2:duration=longest[a]',
        '-map', '0:v',  # видео из первого файла
        '-map', '[a]',  # аудио из фильтра
        '-c:v', 'copy',  # не перекодировать видео
        '-c:a', 'aac',  # кодек аудио
        '-shortest',    # обрезать по самому короткому потоку
        output_path
    ]

    try:
        # Запускаем ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Звук успешно добавлен")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Ошибка при добавлении звука: {e.stderr}")
        # Возвращаем видео без звука
        os.rename(temp_video_path, output_path)
    except FileNotFoundError:
        print("⚠️ ffmpeg не найден. Убедись, что ffmpeg.exe лежит в папке с ботом.")
        os.rename(temp_video_path, output_path)

    # Удаляем временный файл
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())