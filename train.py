import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class DatasetHandler:
    def __init__(self, dataset_dir, train_dir, test_dir, val_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

    # ❌ NOT NEEDED (COMMENTED)
    """
    def download_dataset(self):
        pass

    def unzip_dataset(self):
        pass
    """

    def get_image_dataset_from_directory(self, dir_name):
        dir_path = os.path.join(self.dataset_dir, dir_name)

        return tf.keras.utils.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            color_mode='rgb',
            seed=42,
            batch_size=32,
            image_size=(128, 128)
        )

    def load_split_data(self):
        train_data = self.get_image_dataset_from_directory(self.train_dir)
        test_data = self.get_image_dataset_from_directory(self.test_dir)
        val_data = self.get_image_dataset_from_directory(self.val_dir)

        return train_data, test_data, val_data


class DeepfakeDetectorModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(128, 128, 3)),

            # ✅ FIXED RESCALING
            layers.Rescaling(1./255),

            # ✅ DATA AUGMENTATION
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),

            # ✅ LIGHTWEIGHT CNN (faster for hackathon)
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def compile_model(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    def train_model(self, train_data, val_data, epochs):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
            ModelCheckpoint('best_model.keras', save_best_only=True)
        ]

        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )

    def evaluate_model(self, test_data):
        return self.model.evaluate(test_data)

    def save_model(self, path):
        self.model.save(path)


class TrainModel:
    def __init__(self, dataset_dir, train_dir, test_dir, val_dir):
        self.dataset_handler = DatasetHandler(dataset_dir, train_dir, test_dir, val_dir)

    def run_training(self, learning_rate=0.0001, epochs=20):
        train_data, test_data, val_data = self.dataset_handler.load_split_data()

        model = DeepfakeDetectorModel()
        model.compile_model(learning_rate)

        history = model.train_model(train_data, val_data, epochs)
        evaluation_metrics = model.evaluate_model(test_data)

        model.save_model('deepfake_detector_model.keras')

        return history, evaluation_metrics


if __name__ == '__main__':
    # Relative path — works on any machine
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Dataset')

    train_dir = 'Train'
    test_dir = 'Test'
    val_dir = 'Validation'

    trainer = TrainModel(
        dataset_dir=dataset_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        val_dir=val_dir
    )

    history, evaluation_metrics = trainer.run_training()

    print('evaluation metrics:', evaluation_metrics)