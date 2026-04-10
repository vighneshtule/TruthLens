from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import numpy as np
import cv2
import base64
import os

IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

# EfficientNetB0 was trained at 224×224
IMAGE_SIZE = (224, 224)


class InferenceModel:
    """
    Loads a trained EfficientNetB0-based model and serves predictions via Flask.
    """

    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.model_path = model_path

        @self.app.route('/', methods=['GET', 'POST'])
        def upload_file():
            if request.method == 'POST':
                if 'file' not in request.files:
                    return render_template('index.html', error='No file part in request.')
                file = request.files['file']
                if file.filename == '':
                    return render_template('index.html', error='No file selected.')
                if file and self.allowed_file(file.filename):
                    filename = os.path.join(self.app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filename)
                    media_type = self.get_media_type(file.filename)
                    try:
                        if media_type == 'video':
                            prediction, prediction_percentage, analyzed_frames = self.predict_video(filename)
                            gradcam_img = None
                        else:
                            prediction, prediction_percentage, gradcam_img = self.predict_image(filename)
                            analyzed_frames = 1
                    except ValueError as err:
                        return render_template('index.html', error=str(err))
                    finally:
                        if os.path.exists(filename):
                            os.remove(filename)

                    result = 'Fake' if prediction >= 0.5 else 'Real'
                    return render_template(
                        'index.html',
                        result=result,
                        prediction_percentage=prediction_percentage,
                        gradcam_img=gradcam_img,
                        media_type=media_type,
                        analyzed_frames=analyzed_frames,
                    )
                else:
                    return render_template(
                        'index.html',
                        error='Allowed types: png, jpg, jpeg, mp4, mov, avi, mkv, webm',
                    )
            return render_template('index.html')

    # ── Helpers ──────────────────────────────────────────────────────────────

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)

    def get_media_type(self, filename):
        ext = filename.rsplit('.', 1)[1].lower()
        return 'video' if ext in VIDEO_EXTENSIONS else 'image'

    # ── Image preprocessing ───────────────────────────────────────────────────

    def load_img_array(self, file_path):
        """Load image as (1, 224, 224, 3) float32 array in [0, 255]."""
        img = tf.keras.utils.load_img(file_path, target_size=IMAGE_SIZE)
        arr = tf.keras.utils.img_to_array(img)          # [0, 255]
        return np.expand_dims(arr, axis=0).astype(np.float32)

    # ── Test-Time Augmentation (TTA) ──────────────────────────────────────────

    def tta_predict(self, img_array):
        """
        Average predictions over 4 augmented variants of the image.
        Reduces variance on borderline cases and improves effective accuracy.
        """
        scores = []

        variants = [
            img_array,                                          # original
            img_array[:, :, ::-1, :],                          # horizontal flip
            np.rot90(img_array, k=1, axes=(1, 2)),             # 90° rotation
            np.rot90(img_array, k=3, axes=(1, 2)),             # 270° rotation
        ]

        for v in variants:
            score = float(self.model.predict(v, verbose=0)[0][0])
            scores.append(score)

        return float(np.mean(scores))

    # ── Grad-CAM ───────────────────────────────────────────────────────────────

    def generate_gradcam(self, img_array, file_path):
        """
        Generate a Grad-CAM heatmap overlay for the given image.
        Supports both EfficientNetB0 and legacy scratch CNN models.

        Returns base64-encoded PNG data URI, or None on failure.
        """
        # Find the last Conv2D layer dynamically (works for any architecture)
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
            # Handle EfficientNetB0 sub-model: search inside it
            if hasattr(layer, 'layers'):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv_layer = sub
                        break
                if last_conv_layer:
                    break

        if last_conv_layer is None:
            return None

        # Build a model that outputs (conv activations, final prediction)
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[last_conv_layer.output, self.model.output]
        )

        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            tape.watch(conv_outputs)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        orig = cv2.imread(file_path)
        if orig is None:
            return None
        h, w = orig.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.55, heatmap_colored, 0.45, 0)

        _, buffer = cv2.imencode('.png', overlay)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return f'data:image/png;base64,{img_b64}'

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict_image(self, file_path):
        """
        Predict Real/Fake for an image using TTA.

        Returns:
            tuple: (prediction_score, prediction_percentage, gradcam_b64)
        """
        img_array = self.load_img_array(file_path)
        prediction = self.tta_predict(img_array)
        prediction_percentage = prediction * 100
        gradcam_img = self.generate_gradcam(img_array, file_path)
        return prediction, prediction_percentage, gradcam_img

    def preprocess_video_frame(self, frame):
        """Convert an OpenCV BGR frame to model input tensor (1, 224, 224, 3)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, IMAGE_SIZE)
        return np.expand_dims(frame_resized.astype(np.float32), axis=0)

    def predict_video(self, file_path, max_frames=16):
        """
        Predict deepfake probability for a video by sampling up to `max_frames` frames.

        Returns:
            tuple: (prediction_score, prediction_percentage, analyzed_frames)
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError('Unable to open video file.')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            sample_count = min(max_frames, total_frames)
            sample_indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int)
        else:
            sample_indices = np.array([], dtype=int)

        frame_scores = []

        if sample_indices.size > 0:
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                img_array = self.preprocess_video_frame(frame)
                score = float(self.model.predict(img_array, verbose=0)[0][0])
                frame_scores.append(score)
        else:
            # Fallback: stride-based sampling when frame count is unavailable
            stride, i = 10, 0
            while len(frame_scores) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    img_array = self.preprocess_video_frame(frame)
                    score = float(self.model.predict(img_array, verbose=0)[0][0])
                    frame_scores.append(score)
                i += 1

        cap.release()

        if not frame_scores:
            raise ValueError('Unable to decode frames from video.')

        prediction = float(np.mean(frame_scores))
        prediction_percentage = prediction * 100
        return prediction, prediction_percentage, len(frame_scores)

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    model_path = 'deepfake_detector_model.keras'
    inference_model = InferenceModel(model_path)
    inference_model.run()
