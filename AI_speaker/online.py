import ollama
import os
import time
import vosk
import json
import pyaudio
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# Инициализация Ollama
ollama_client = ollama.Client()

# Инициализация VOSK
vosk_model_path = "models/vosk-model-small-ru-0.22"
#vosk_model_path = "models/vosk-model-small-en-us-0.15"
vosk_model = vosk.Model(vosk_model_path)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

# Инициализация PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.stop_stream()  # Начальное состояние - микрофон выключен

# Функция для чтения системного промпта
def read_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# Функция для генерации ответа
def generate_response(prompt, system_prompt):
    response = ollama_client.chat(model='llama3.1', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ])
    return response

# Функция для синтеза речи и воспроизведения ответа
def speak(text):
    tts = gTTS(text=text, lang='ru')
    tts.save("response.mp3")
    audio = AudioSegment.from_mp3("response.mp3")
    play(audio)

# Функция для голосового ввода и распознавания речи
def listen():
    # Озвучиваем сообщение "Говорите"

    print("Слушаю...")
    frames = []
    silence_threshold = 50000
    speech_threshold = 1000
    max_recording_time = 3
    start_time = time.time()
    speech_detected = False

    # Включаем микрофон
    stream.start_stream()
    print("Микрофон включен")

    # Очистка аудиопотока перед началом записи
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_data).mean() < silence_threshold:
            break

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)

        if not speech_detected:
            if np.abs(audio_data).mean() > speech_threshold:
                speech_detected = True
                start_time = time.time()
        else:
            if np.abs(audio_data).mean() < silence_threshold:
                if time.time() - start_time > 1:
                    break
            if time.time() - start_time > max_recording_time:
                break

    # Выключаем микрофон
    stream.stop_stream()
    print("Микрофон выключен")

    audio_data = b''.join(frames)
    if recognizer.AcceptWaveform(audio_data):
        result = recognizer.Result()
        text = json.loads(result)["text"]
        print(f"Вы сказали: {text}")
        return text
    return None

# Основная функция
def main():
    system_prompt_file = "kringe.txt"
    system_prompt = read_system_prompt(system_prompt_file)

    try:
        while True:
            user_input = listen()
            if user_input:
                response = generate_response(user_input, system_prompt)
                print(f"Ollama: {response}")
                response_text = response['message']['content']
                time.sleep(1)  # Небольшая задержка перед ответом
                speak(response_text)
    except KeyboardInterrupt:
        print("Завершение работы...")
    finally:
        # Освобождаем ресурсы перед завершением работы скрипта
        stream.close()
        p.terminate()
        recognizer.__del__()

if __name__ == "__main__":
    main()