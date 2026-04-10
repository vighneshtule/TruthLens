import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ── Mixed precision for ~2× GPU speedup ──────────────────────────────────────
tf.keras.mixed_precision.set_global_policy('mixed_float16')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


class DatasetHandler:
    def __init__(self, dataset_dir, train_dir, test_dir, val_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

    def get_image_dataset_from_directory(self, dir_name, augment=False):
        dir_path = os.path.join(self.dataset_dir, dir_name)

        ds = tf.keras.utils.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            color_mode='rgb',
            seed=42,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
        )

        if augment:
            augmentation = tf.keras.Sequential([
                layers.RandomFlip('horizontal'),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.2),
                layers.RandomBrightness(0.2),
            ])
            ds = ds.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        return ds.prefetch(tf.data.AUTOTUNE)

    def load_split_data(self):
        train_data = self.get_image_dataset_from_directory(self.train_dir, augment=True)
        test_data = self.get_image_dataset_from_directory(self.test_dir, augment=False)
        val_data = self.get_image_dataset_from_directory(self.val_dir, augment=False)
        return train_data, test_data, val_data


class DeepfakeDetectorModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        # ── EfficientNetB0 backbone (ImageNet pretrained) ──────────────────
        backbone = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMAGE_SIZE, 3),
        )
        backbone.trainable = False  # frozen for Phase 1

        inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

        # EfficientNetB0 expects pixels in [0, 255] — it handles rescaling internally
        x = backbone(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        # dtype=float32 on final layer to avoid float16 softmax instability
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

        model = tf.keras.Model(inputs, outputs)
        return model

    def compile_model(self, learning_rate, label_smoothing=0.1):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    def unfreeze_top_layers(self, num_layers=30):
        """Unfreeze the top N layers of the EfficientNetB0 backbone for fine-tuning."""
        backbone = self.model.layers[1]  # EfficientNetB0 is the second layer
        backbone.trainable = True
        for layer in backbone.layers[:-num_layers]:
            layer.trainable = False
        print(f'Unfrozen top {num_layers} layers of EfficientNetB0 for fine-tuning.')

    def train_model(self, train_data, val_data, epochs, model_path='deepfake_detector_model.keras'):
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7),
            ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        ]
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
        )

    def evaluate_model(self, test_data):
        return self.model.evaluate(test_data)

    def save_model(self, path):
        self.model.save(path)


class TrainModel:
    def __init__(self, dataset_dir, train_dir, test_dir, val_dir):
        self.dataset_handler = DatasetHandler(dataset_dir, train_dir, test_dir, val_dir)

    def run_training(self):
        train_data, test_data, val_data = self.dataset_handler.load_split_data()

        model = DeepfakeDetectorModel()

        # ── Phase 1: Train classifier head only (backbone frozen) ────────
        print('\n=== Phase 1: Training classifier head (backbone frozen) ===')
        model.compile_model(learning_rate=1e-3, label_smoothing=0.1)
        history_phase1 = model.train_model(train_data, val_data, epochs=10)

        # ── Phase 2: Fine-tune top 30 EfficientNetB0 layers ─────────────
        print('\n=== Phase 2: Fine-tuning top 30 EfficientNetB0 layers ===')
        model.unfreeze_top_layers(num_layers=30)
        model.compile_model(learning_rate=1e-5, label_smoothing=0.0)
        history_phase2 = model.train_model(train_data, val_data, epochs=20)

        print('\n=== Final Evaluation on Test Set ===')
        evaluation_metrics = model.evaluate_model(test_data)
        print('Test metrics (loss, accuracy, precision, recall):', evaluation_metrics)

        return history_phase1, history_phase2, evaluation_metrics


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Dataset')

    train_dir = 'Train'
    test_dir = 'Test'
    val_dir = 'Validation'

    trainer = TrainModel(
        dataset_dir=dataset_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        val_dir=val_dir,
    )

    trainer.run_training()