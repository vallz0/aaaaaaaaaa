import pickle
import random
import numpy as np
import sounddevice as sd
from TTS.api import TTS

def fale(mensagem, model_path="voice_model.pkl"):
    try:
        with open(model_path, "rb") as f:
            voice_data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Modelo não encontrado! Execute 'train_voice.py' primeiro.")
        return

    audio_files = voice_data.get("audio_files", [])
    if not audio_files:
        print("⚠️ Nenhum áudio foi treinado.")
        return

    audio_path = random.choice(audio_files)
    print(f"🔊 Usando o áudio: {audio_path}")

    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=False)

    wav = tts.tts(text=mensagem, speaker_wav=audio_path, language="pt-br")

    wav_np = np.array(wav, dtype=np.float32)
    sd.default.device = 0
    sd.play(wav_np, samplerate=16000)
    sd.wait()

if __name__ == '__main__':

    while True:
        a = str(input())
        if a == "sair do loop":
            break
        else:
            fale(a)

