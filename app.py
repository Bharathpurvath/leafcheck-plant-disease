from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title='LeafCheck | Plant disease assistant',page_icon='🌿',layout='centered')
st.title('🌿 LeafCheck')
st.caption('Plant leaf image screening with deep learning')
st.warning('A model prediction is a screening aid, not a professional diagnosis or pesticide instruction. Confirm with a local agricultural extension specialist before acting.')
MODEL=Path('models/plant_disease.keras'); LABELS=MODEL.with_name('class_names.json')
@st.cache_resource
def load_model(): return tf.keras.models.load_model(MODEL)
if not MODEL.exists() or not LABELS.exists():
 st.info('No trained model is bundled yet. Follow README.md to train on the PlantVillage dataset, then place the model and class_names.json in models/.')
 st.stop()
names=json.loads(LABELS.read_text(encoding='utf-8')); model=load_model()
file=st.file_uploader('Upload a clear leaf photo',type=['jpg','jpeg','png'])
if file:
 image=Image.open(file).convert('RGB'); st.image(image,caption='Uploaded leaf',use_container_width=True)
 arr=np.asarray(image.resize((224,224)),dtype=np.float32)[None,...]
 with st.spinner('Analyzing image…'):
  probs=model.predict(arr,verbose=0)[0]; idx=int(np.argmax(probs)); confidence=float(probs[idx])
 st.subheader(names[idx].replace('___',' — ').replace('_',' ')); st.metric('Model confidence',f'{confidence:.1%}')
 if confidence < .60: st.warning('Confidence is low. Try a sharper, well-lit close-up with one leaf in focus.')
 st.caption('Top predictions')
 for i in np.argsort(probs)[-3:][::-1]: st.write(f"{names[int(i)].replace('___',' — ').replace('_',' ')}: {probs[i]:.1%}")
 st.divider(); st.subheader('Model attention (Grad-CAM)')
 try:
  convs=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D)]
  # The transfer backbone is nested in the functional model.
  nested=[l for l in model.layers if isinstance(l,tf.keras.Model)]
  convs += [l for m in nested for l in m.layers if isinstance(l,tf.keras.layers.Conv2D)]
  last=convs[-1]; backbone=next((l for l in nested if last.name in [x.name for x in l.layers]),None)
  if backbone is None: raise ValueError('Could not locate convolution layer')
  x=tf.convert_to_tensor(arr); x=model.get_layer('augmentation')(x,training=False); x=model.get_layer('rescaling')(x)
  feature_net=tf.keras.Model(backbone.input,[backbone.get_layer(last.name).output,backbone.output])
  base_index=model.layers.index(backbone)
  with tf.GradientTape() as tape:
   conv_out,base_out=feature_net(x,training=False); predictions=base_out
   for layer in model.layers[base_index+1:]: predictions=layer(predictions,training=False)
   score=predictions[:,idx]
  grads=tape.gradient(score,conv_out); weights=tf.reduce_mean(grads,axis=(1,2)); cam=tf.reduce_sum(conv_out*weights[:,None,None,:],axis=-1)[0]; cam=tf.maximum(cam,0); cam/=tf.maximum(tf.reduce_max(cam),1e-8); heat=tf.image.resize(cam[...,None],(224,224)).numpy().squeeze()
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(); ax.imshow(image.resize((224,224))); ax.imshow(heat,cmap='jet',alpha=.38); ax.axis('off'); st.pyplot(fig); plt.close(fig)
 except Exception as e: st.caption(f'Grad-CAM is unavailable for this model build ({e}).')



