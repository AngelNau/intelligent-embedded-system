import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras # type: ignore
import tensorflow_model_optimization as tfmot
import tf2onnx
import onnx


def representative_data_gen():
    for image_batch, _ in train_ds.take(100):  # type: ignore
        yield [image_batch]


def create_tiny_cnn(num_classes=10, input_shape=(128, 128, 3)):
    inputs = keras.Input(shape=input_shape, name='input')
    x = keras.layers.Conv2D(16, (3, 3), padding='same', name='conv1')(inputs)
    x = keras.layers.ReLU(name='relu1')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', name='conv2')(x)
    x = keras.layers.ReLU(name='relu2')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(num_classes, name='dense')(x)
    outputs = keras.layers.Softmax(name='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='TinyCNN')
    return model


def train_and_get_best_model(model, train_ds, val_ds, epochs=30, learning_rate=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    temp_model_path = "temp_best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "temp_best_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    print(f"Training for up to {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    best_val_acc = max(history.history['val_accuracy'])
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model, history

data_dir = "<dataset_dir>"
seed = 42

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    batch_size=32,
    image_size=(128, 128)
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    batch_size=32,
    image_size=(128, 128)
)

# Create model
model = create_tiny_cnn(num_classes=5, input_shape=(128, 128, 3))
model.build(input_shape=(None, 128, 128, 3))  # CIFAR-10 input shape
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

best_model, history = train_and_get_best_model(model, train_ds, val_ds, epochs=1)

# Save to ONNX
# input_signature = [tf.TensorSpec([1, 128, 128, 3], tf.uint8, name='input')] # type: ignore
# onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=11)
# onnx.save(onnx_model, "tinyCnn_tf.onnx")

# Apply quantization aware training
# q_aware_model = tfmot.quantization.keras.quantize_model(best_model)
# q_aware_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # type: ignore
converter.representative_dataset = representative_data_gen # type: ignore
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
quantized_tflite_model = converter.convert()

# Save the model
import pathlib
path = pathlib.Path("./tiny_cnn.tflite")
path.write_bytes(quantized_tflite_model) # type: ignore

print("Model successfully quantized and exported to TFLite!")
