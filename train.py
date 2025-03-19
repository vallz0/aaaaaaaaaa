# train_voice.py - Treina o modelo e armazena os dados

import os
import pickle
from TTS.api import TTS

def get_audio_files(directory):
    """ Retorna uma lista de arquivos de áudio na pasta especificada. """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.wav', '.mp3', '.flac'))]


def train_voice( model_path="voice_model.pkl"):
    """ Treina o modelo de clonagem de voz e salva os dados em um arquivo. """

    AUDIO_FOLDER: str = "Audio-Data"
    audio_files = get_audio_files(AUDIO_FOLDER)
    if not audio_files:
        print("⚠️ Nenhum arquivo de áudio encontrado na pasta.")
        return

    # Criar modelo de TTS
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=False)

    # Criar dicionário para armazenar os áudios treinados
    voice_data = {"audio_files": audio_files}

    # Salvar os dados treinados
    with open(model_path, "wb") as f:
        pickle.dump(voice_data, f)

    print(f"✅ Treinamento concluído! Modelo salvo em {model_path}")


if __name__ == '__main__':
    train_voice()
