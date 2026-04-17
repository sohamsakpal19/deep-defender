import librosa
import numpy as np

from src.models.hf_audio_model import HFAudioModel
from src.utils.postprocess import map_probability_to_label

class AudioPipeline:
    def __init__(self, model: HFAudioModel, preprocess_fn=None, default_sr=16000):
        """
        preprocess_fn: optional callable that returns (waveform, sample_rate).
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.default_sr = default_sr

    def _load_audio(self, audio_path):
        # Handles any audio file supported by librosa
        waveform, sr = librosa.load(audio_path, sr=self.default_sr, mono=True)
        return waveform, sr

    def run(self, audio_path=None, waveform=None, sample_rate=None):
        """
        audio_path: path to audio file
        waveform: 1D numpy array (optional)
        sample_rate: int (optional)
        """
        if waveform is None:
            if self.preprocess_fn:
                waveform, sample_rate = self.preprocess_fn(audio_path)
            else:
                waveform, sample_rate = self._load_audio(audio_path)

        # Ensure numpy float32
        if isinstance(waveform, np.ndarray):
            waveform = waveform.astype(np.float32)

        logits, probs = self.model.predict(waveform, sample_rate)

        # Assume index 0 = Fake, 1 = Real (common in binary HF classifiers)
        p_fake = float(probs[0].item())
        label = map_probability_to_label(p_fake)

        return {
            "label": label,
            "p_fake": p_fake,
            "probs": probs.detach().cpu().tolist(),
            "id2label": self.model.id2label()
        }