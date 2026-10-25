from pathlib import Path
import argparse, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def main():
 p=argparse.ArgumentParser(description='Create a Grad-CAM overlay for one leaf image')
 p.add_argument('--image',required=True); p.add_argument('--model',default='models/plant_disease.keras'); p.add_argument('--output',default='reports/gradcam_example.png'); a=p.parse_args()
 model=tf.keras.models.load_model(a.model); names=json.loads(Path(a.model).with_name('class_names.json').read_text(encoding='utf-8'))
 image=Image.open(a.image).convert('RGB'); resized=image.resize((224,224)); arr=np.asarray(resized,dtype=np.float32)[None,...]
 probs=model.predict(arr,verbose=0)[0]; idx=int(probs.argmax()); nested=[layer for layer in model.layers if isinstance(layer,tf.keras.Model)]
 backbone=next((layer for layer in nested if any(isinstance(x,tf.keras.layers.Conv2D) for x in layer.layers)),None)
 if backbone is None: raise RuntimeError('Could not locate convolutional backbone')
 last=[layer for layer in backbone.layers if isinstance(layer,tf.keras.layers.Conv2D)][-1]
 arr_tensor=tf.convert_to_tensor(arr); x=model.get_layer('augmentation')(arr_tensor,training=False); x=model.get_layer('rescaling')(x)
 feature_net=tf.keras.Model(backbone.input,[backbone.get_layer(last.name).output,backbone.output]); base_index=model.layers.index(backbone)
 with tf.GradientTape() as tape:
  fmap,base_out=feature_net(x,training=False); y=base_out
  for layer in model.layers[base_index+1:]: y=layer(y,training=False)
  score=y[:,idx]
 grads=tape.gradient(score,fmap)
 if grads is None: raise RuntimeError('Gradient could not be computed for the selected class')
 weights=tf.reduce_mean(grads,axis=(1,2)); cam=tf.reduce_sum(fmap*weights[:,None,None,:],axis=-1)[0]; cam=tf.maximum(cam,0); cam/=tf.maximum(tf.reduce_max(cam),1e-8); heat=tf.image.resize(cam[...,None],(224,224)).numpy().squeeze()
 fig,ax=plt.subplots(figsize=(7,7)); ax.imshow(resized); ax.imshow(heat,cmap='jet',alpha=.42); ax.axis('off'); ax.set_title(f"{names[idx].replace('___',' — ').replace('_',' ')} | confidence {probs[idx]:.1%}"); fig.tight_layout(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=180,bbox_inches='tight'); plt.close(fig); print(f'Predicted: {names[idx]} ({probs[idx]:.4f}); saved: {out}')
if __name__=='__main__':main()


