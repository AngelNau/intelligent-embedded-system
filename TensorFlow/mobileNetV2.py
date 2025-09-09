import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import tf2onnx
import onnx


def representative_data_gen():
    for image_batch, _ in train_ds.take(100):  # type: ignore
        yield [image_batch]


def create_mobilenet_for_quantization(num_classes=5):
    base_model = keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        alpha=0.35,
        weights=None, # type: ignore
        include_top=True,
        pooling='avg',
        classes=num_classes
    )
    # Use MobileNetV2 architecture with pre-trained weights on ImageNet
    backbone = keras.applications.MobileNetV2((128, 128, 3), alpha=0.35, weights="imagenet", include_top=False)
    # Transfer the weights from the pre-trained MobileNetV2 model to the random model
    for i, layer in enumerate(backbone.layers):
        base_model.layers[i].set_weights(layer.get_weights())
    return base_model

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

data_dir = "../flower_photos/"
model_name = "mobileNetV2"
seed = 42

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=seed,
    batch_size=32,
    image_size=(128, 128)
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=seed,
    batch_size=32,
    image_size=(128, 128)
)

model = create_mobilenet_for_quantization()
best_model, history = train_and_get_best_model(model, train_ds, val_ds, epochs=1)

floatConverter = tf.lite.TFLiteConverter.from_keras_model(best_model)
floatConverter.inference_input_type = tf.uint8

quantConverter = tf.lite.TFLiteConverter.from_keras_model(best_model)
quantConverter.optimizations = [tf.lite.Optimize.DEFAULT] # type: ignore
quantConverter.representative_dataset = representative_data_gen # type: ignore
quantConverter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
quantConverter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

# input_signature = [tf.TensorSpec([1, 128, 128, 3], tf.uint8, name='input')] # type: ignore
# onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=11)
# onnx.save(onnx_model, "dst/path/model.onnx")
quant_tflite_model = quantConverter.convert()
float_tflite_model = floatConverter.convert()
# Save the model
import pathlib
path = pathlib.Path(f"./float_{model_name}.tflite")
path.write_bytes(float_tflite_model) # type: ignore
path = pathlib.Path(f"./quantized_{model_name}.tflite")
path.write_bytes(quant_tflite_model) # type: ignore

print("Model successfully quantized and exported to TFLite!")
