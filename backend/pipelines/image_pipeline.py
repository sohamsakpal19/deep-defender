from PIL import Image
import numpy as np

from src.models.hf_image_model import HFImageModel
from src.utils.postprocess import map_probability_to_label

class ImagePipeline:
    def __init__(self, model: HFImageModel, preprocess_fn=None):
        """
        preprocess_fn: optional callable that returns a PIL.Image or RGB numpy array.
        """
        self.model = model
        self.preprocess_fn = preprocess_fn

    def _load_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def run(self, image_path=None, image=None):
        """
        image_path: path to image (optional)
        image: PIL.Image or RGB numpy array (optional)
        """
        if image is None:
            if self.preprocess_fn:
                image = self.preprocess_fn(image_path)
            else:
                image = self._load_image(image_path)

        # Ensure numpy arrays are uint8 RGB
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

        logits, probs = self.model.predict(image)

        # Labels: 0=Fake, 1=Real (from model card)
        p_fake = float(probs[0].item())
        label = map_probability_to_label(p_fake)

        return {
            "label": label,
            "p_fake": p_fake,
            "probs": probs.detach().cpu().tolist(),
            "id2label": self.model.id2label()
        }