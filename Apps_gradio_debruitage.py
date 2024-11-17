import gradio as gr
import torch
from demucs import pretrained
from demucs.apply import apply_model
import torchaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile as sf
import tempfile
import uuid
import os

# Charger le modèle pré-entraîné de Demucs
def load_model():
    try:
        model = pretrained.get_model('mdx_extra')
        model.cpu()
        return model
    except Exception as e:
        raise gr.Error(f"Erreur lors du chargement du modèle : {str(e)}")

model = load_model()

# Fonction de débruitage
def debruitage(audio_file):
    try:
        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.unsqueeze(0)
        
        sources = apply_model(model, waveform, split=True)
        vocal_source = sources[0, 3, :, :]
        vocal_source = torch.mean(vocal_source, dim=0)
        
        temp_dir = tempfile.gettempdir()
        isolated_voice_path = f"{temp_dir}/voix_isolee_{uuid.uuid4().hex}.wav"
        sf.write(isolated_voice_path, vocal_source.cpu().numpy(), sr)
        
        return isolated_voice_path
    except Exception as e:
        raise gr.Error(f"Erreur lors du débruitage : {str(e)}")

# Fonction de détection de bruit
def detectage_bruit(audio_file):
    try:
        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.unsqueeze(0)
        
        sources = apply_model(model, waveform, split=True)
        noise_source = sources[0, 2, :, :].cpu().numpy()
        noise_source = np.mean(noise_source, axis=0)
        
        frame_size = 1024
        hop_length = 512
        rms = librosa.feature.rms(y=noise_source, frame_length=frame_size, hop_length=hop_length).flatten()
        
        threshold = np.mean(rms) + 1.5 * np.std(rms)
        in_noise = False
        noise_intervals = []
        for i, energy in enumerate(rms):
            if energy > threshold and not in_noise:
                start_time = i * hop_length / sr
                in_noise = True
            elif energy <= threshold and in_noise:
                end_time = i * hop_length / sr
                noise_intervals.append((start_time, end_time))
                in_noise = False
        
        if in_noise:
            noise_intervals.append((start_time, len(noise_source) / sr))
        
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(waveform.squeeze().numpy(), sr=sr, alpha=0.5, color="blue", ax=ax)
        for start, end in noise_intervals:
            ax.axvspan(start, end, color="red", alpha=0.3)
        ax.set_xlabel("Temps (secondes)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Détection des intervalles de bruit continu")
        
        return fig, f"Nombre de segments de bruit détectés : {len(noise_intervals)}."
    except Exception as e:
        raise gr.Error(f"Erreur lors de la détection de bruit : {str(e)}")

# Fonction de correction rapide
def correction_rapide(audio_file):
    try:
        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.unsqueeze(0)
        
        sources = apply_model(model, waveform, split=True)
        vocal_source = sources[0, 3, :, :]
        vocal_source = torch.mean(vocal_source, dim=0)
        
        temp_dir = tempfile.gettempdir()
        isolated_voice_path = f"{temp_dir}/voix_isolee_{uuid.uuid4().hex}.wav"
        sf.write(isolated_voice_path, vocal_source.cpu().numpy(), sr)
        
        voice, sr = sf.read(isolated_voice_path)
        sos = signal.butter(10, 1500, 'lp', fs=sr, output='sos')
        filtered_voice = signal.sosfilt(sos, voice)
        filtered_voice = np.int16(filtered_voice / np.max(np.abs(filtered_voice)) * 32767)
        
        filtered_voice_path = f"{temp_dir}/voix_filtre_{uuid.uuid4().hex}.wav"
        sf.write(filtered_voice_path, filtered_voice, sr)
        
        return filtered_voice_path
    except Exception as e:
        raise gr.Error(f"Erreur lors de la correction rapide : {str(e)}")

# Interface Gradio
with gr.Blocks(title="Traitement Audio") as demo:
    gr.Markdown("# 🎵 Traitement Audio : Détection et Débruitage")
    gr.Markdown("Bienvenue dans notre application de traitement audio. Téléchargez un fichier audio pour détecter ou réduire les bruits.")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Accueil"):
            gr.Markdown("Utilisez les onglets pour accéder aux différentes fonctionnalités.")
            gr.Image("image_audio.jpeg", label="Image d'accueil")
        
        with gr.TabItem("Correction rapide"):
            with gr.Row():
                audio_input_correction = gr.Audio(label="Téléchargez un fichier audio (MP3 ou WAV)", type="filepath")
                audio_output_correction = gr.Audio(label="Voix filtrée")
            correction_button = gr.Button("Corriger")
            correction_button.click(correction_rapide, inputs=audio_input_correction, outputs=audio_output_correction)
        
        with gr.TabItem("Détectage de Bruit"):
            with gr.Row():
                audio_input_detection = gr.Audio(label="Téléchargez un fichier audio (MP3 ou WAV)", type="filepath")
                plot_output = gr.Plot(label="Segments de Bruit Détectés")
            detection_info = gr.Textbox(label="Résultat de la détection")
            detection_button = gr.Button("Détecter le bruit")
            detection_button.click(detectage_bruit, inputs=audio_input_detection, outputs=[plot_output, detection_info])
        
        with gr.TabItem("Débruitage"):
            with gr.Row():
                audio_input_debruitage = gr.Audio(label="Téléchargez un fichier audio (MP3 ou WAV)", type="filepath")
                audio_output_debruitage = gr.Audio(label="Voix isolée")
            debruitage_button = gr.Button("Débruiter")
            debruitage_button.click(debruitage, inputs=audio_input_debruitage, outputs=audio_output_debruitage)

demo.launch(share=True)