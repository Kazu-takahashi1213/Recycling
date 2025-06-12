"""
"C:\Users\kazu1\Pictures\Screenshots\スクリーンショット 2025-06-12 025144.png"
"""

import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 元のデータセット（Kaggle）フォルダ
RAW_DATASET = "dataset"

# マッピング後のフォルダ
PROCESSED_DATASET = "processed_dataset"

# カテゴリ変換マップ
category_map = {
    "biological": "biowaste",
    "paper": "paper",
    "cardboard": "paper",
    "plastic": "wertstoff",
    "metal": "wertstoff",
    "glass": "glass",
    "trash": "residual",
    "clothes": "residual",
    "shoes": "residual"
    # battery は除外
}

# 変換されたデータ構造を再構築
if os.path.exists(PROCESSED_DATASET):
    shutil.rmtree(PROCESSED_DATASET)
os.makedirs(PROCESSED_DATASET)

for raw_cat, new_cat in category_map.items():
    raw_dir = os.path.join(RAW_DATASET, raw_cat)
    new_dir = os.path.join(PROCESSED_DATASET, new_cat)
    os.makedirs(new_dir, exist_ok=True)

    if os.path.exists(raw_dir):
        for file in os.listdir(raw_dir):
            src = os.path.join(raw_dir, file)
            dst = os.path.join(new_dir, file)
            shutil.copyfile(src, dst)

# 転移学習（MobileNetV2）設定
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "model/classifier_kaggle_paderborn.h5"

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    PROCESSED_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    PROCESSED_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

print(f"\nモデル訓練完了！保存先: {MODEL_PATH}")
