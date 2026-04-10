from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import tensorflow as tf
import numpy as np
import cv2
import base64
import os

IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

class InferenceModel:
    """
    A class to load a trained model and handle file uploads for predictions.
    """

    def __init__(self, model_path):
        """
        Initialize the InferenceModel class.

        Args:
            model_path (str): Path to the saved Keras model.
        """
        self.model = load_model(model_path)
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.model_path = model_path

        @self.app.route('/', methods=['GET', 'POST'])
        def upload_file():
            """
            Handle file upload and prediction requests.

            Returns:
            --------
            str
                The rendered HTML template with the result or error message.
            """
            if request.method == 'POST':
                # check if the post request has the file part
                if 'file' not in request.files:
                    return render_template('index.html', error='no file part')
                file = request.files['file']
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    return render_template('index.html', error='no selected file')
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
                        analyzed_frames=analyzed_frames
                    )
                else:
                    return render_template(
                        'index.html',
                        error='allowed file types: png, jpg, jpeg, mp4, mov, avi, mkv, webm'
                    )
            return render_template('index.html')

    def allowed_file(self, filename):
        """
        Check if a file has an allowed extension.

        Parameters:
        -----------
        filename : str
            The name of the file to check.

        Returns:
        --------
        bool
            True if the file has an allowed extension, False otherwise.
        """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)

    def get_media_type(self, filename):
        """
        Identify uploaded media type based on extension.
        """
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return 'video'
        return 'image'

    def generate_gradcam(self, img_array, file_path):
        """
        Generate a Grad-CAM heatmap overlay for the given image.

        Parameters:
        -----------
        img_array : np.ndarray
            Preprocessed image array (1, 128, 128, 3).
        file_path : str
            Path to the original image file for overlay.

        Returns:
        --------
        str
            Base64-encoded PNG data URI of the heatmap overlay.
        """
        # Find the last Conv2D layer dynamically
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return None

        # Run forward pass layer-by-layer, watching the last conv output.
        # This avoids the .output attribute issue on Sequential models.
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            x = img_tensor
            conv_outputs = None
            for layer in self.model.layers:
                x = layer(x)
                if layer is last_conv_layer:
                    conv_outputs = x
                    tape.watch(conv_outputs)
            predictions = x
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight activations by gradients and collapse to single heatmap
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Load original image and overlay heatmap
        orig = cv2.imread(file_path)
        h, w = orig.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.55, heatmap_colored, 0.45, 0)

        # Encode to base64 PNG for embedding directly in HTML
        _, buffer = cv2.imencode('.png', overlay)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_b64}"

    def predict_image(self, file_path):
        """
        Predict whether an image is Real or Fake using the loaded model,
        and generate a Grad-CAM heatmap overlay.

        Parameters:
        -----------
        file_path : str
            The path to the image file.

        Returns:
        --------
        tuple
            A tuple containing the prediction, prediction percentage, and
            base64-encoded Grad-CAM overlay image.
        """
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        result = self.model.predict(img_array)
        prediction = result[0][0]
        prediction_percentage = prediction * 100
        gradcam_img = self.generate_gradcam(img_array, file_path)
        return prediction, prediction_percentage, gradcam_img

    def preprocess_video_frame(self, frame):
        """
        Convert OpenCV BGR frame to model input tensor.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (128, 128))
        img_array = np.expand_dims(frame_resized.astype(np.float32), axis=0)
        return img_array

    def predict_video(self, file_path, max_frames=16):
        """
        Predict deepfake probability for video by sampling frames.

        Returns:
            tuple: (prediction_score, prediction_percentage, analyzed_frames)
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError('unable to open video file')

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
            # fallback for videos where frame count is unavailable
            stride = 10
            i = 0
            while True and len(frame_scores) < max_frames:
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
            raise ValueError('unable to decode frames from video')

        prediction = float(np.mean(frame_scores))
        prediction_percentage = prediction * 100
        return prediction, prediction_percentage, len(frame_scores)

    def run(self):
        """
        Run the Flask application with the loaded model.
        """
        self.app.run(debug=True)


if __name__ == '__main__':
    # inference
    model_path = 'deepfake_detector_model.keras'
    inference_model = InferenceModel(model_path)
    inference_model.run()
