from pathlib import Path
import json, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE=(224,224); BATCH=32; SEED=42

def main():
    p=argparse.ArgumentParser(description='Train MobileNetV2 on a class-folder PlantVillage dataset')
    p.add_argument('--data', required=True, help='Root folder with one subfolder per class')
    p.add_argument('--output', default='models/plant_disease.keras')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--fine-tune-epochs', type=int, default=5)
    p.add_argument('--resume', action='store_true', help='Continue training from --output')
    p.add_argument('--phase', choices=['all', 'head', 'fine-tune'], default='all')
    args=p.parse_args(); root=Path(args.data)
    if not root.is_dir(): raise SystemExit(f'Dataset folder not found: {root}')
    train=keras.utils.image_dataset_from_directory(root, validation_split=.2, subset='training', seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
    val=keras.utils.image_dataset_from_directory(root, validation_split=.2, subset='validation', seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
    names=train.class_names
    if len(names)<2: raise SystemExit('Need at least two class folders.')
    autotune=tf.data.AUTOTUNE
    if args.resume:
        model=keras.models.load_model(args.output)
        saved_names=json.loads(Path(args.output).with_name('class_names.json').read_text(encoding='utf-8'))
        if names != saved_names: raise SystemExit('Dataset class names differ from the saved model class list.')
        base=next((layer for layer in model.layers if isinstance(layer,keras.Model)),None)
        if base is None: raise SystemExit('Could not find the transfer-learning backbone in the saved model.')
    else:
        augment=keras.Sequential([layers.RandomFlip('horizontal'), layers.RandomRotation(.08), layers.RandomZoom(.12), layers.RandomContrast(.12)], name='augmentation')
        base=keras.applications.MobileNetV2(input_shape=(*IMG_SIZE,3), include_top=False, weights='imagenet')
        base.trainable=False
        inputs=keras.Input((*IMG_SIZE,3)); x=augment(inputs); x=layers.Rescaling(1./127.5, offset=-1)(x); x=base(x, training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(.25)(x); outputs=layers.Dense(len(names), activation='softmax')(x)
        model=keras.Model(inputs,outputs)
    train=train.prefetch(autotune); val=val.prefetch(autotune)
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True), keras.callbacks.ReduceLROnPlateau(patience=2), keras.callbacks.ModelCheckpoint('best.keras', save_best_only=True)]
    if args.phase in ('all','head'):
        base.trainable=False
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if args.epochs: model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)
    if args.phase in ('all','fine-tune'):
        base.trainable=True
        for layer in base.layers[:-30]: layer.trainable=False
        model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if args.fine_tune_epochs: model.fit(train, validation_data=val, epochs=args.fine_tune_epochs, callbacks=callbacks)
    dest=Path(args.output); dest.parent.mkdir(parents=True,exist_ok=True); model.save(dest)
    dest.with_name('class_names.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
    print(f'Saved model: {dest}\nClasses: {len(names)}')
if __name__=='__main__': main()


