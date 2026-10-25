from pathlib import Path
import argparse, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True,help='Held-out class-folder test dataset'); p.add_argument('--model',default='models/plant_disease.keras'); p.add_argument('--output',default='reports'); a=p.parse_args()
 model=tf.keras.models.load_model(a.model); names=json.loads(Path(a.model).with_name('class_names.json').read_text(encoding='utf-8'))
 ds=tf.keras.utils.image_dataset_from_directory(a.data,image_size=(224,224),batch_size=32,shuffle=False,class_names=names)
 y=np.concatenate([labels.numpy() for _,labels in ds]); prob=model.predict(ds,verbose=1); pred=prob.argmax(axis=1); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 n=len(names); cm=np.zeros((n,n),dtype=np.int64); np.add.at(cm,(y,pred),1)
 support=cm.sum(axis=1); predicted=cm.sum(axis=0); tp=np.diag(cm); precision=np.divide(tp,predicted,out=np.zeros(n,dtype=float),where=predicted!=0); recall=np.divide(tp,support,out=np.zeros(n,dtype=float),where=support!=0); f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(n,dtype=float),where=(precision+recall)!=0)
 accuracy=float(tp.sum()/max(1,support.sum())); macro=np.array([precision.mean(),recall.mean(),f1.mean()]); weighted=np.array([(precision*support).sum()/support.sum(),(recall*support).sum()/support.sum(),(f1*support).sum()/support.sum()])
 lines=[f"{'class':52} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}"]
 for i,name in enumerate(names): lines.append(f'{name[:52]:52} {precision[i]:9.4f} {recall[i]:9.4f} {f1[i]:9.4f} {support[i]:9d}')
 lines += ['',f'{"accuracy":52} {"":>9} {"":>9} {accuracy:9.4f} {support.sum():9d}',f'{"macro avg":52} {macro[0]:9.4f} {macro[1]:9.4f} {macro[2]:9.4f} {support.sum():9d}',f'{"weighted avg":52} {weighted[0]:9.4f} {weighted[1]:9.4f} {weighted[2]:9.4f} {support.sum():9d}']
 (out/'classification_report.txt').write_text('\n'.join([f'Test images: {support.sum()}',f'Accuracy: {accuracy:.4f}','',*lines]),encoding='utf-8')
 fig,ax=plt.subplots(figsize=(16,14)); im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax); ax.set(xticks=range(n),yticks=range(n),xticklabels=names,yticklabels=names,xlabel='Predicted',ylabel='True',title='Plant disease confusion matrix'); plt.setp(ax.get_xticklabels(),rotation=90,ha='right',fontsize=6); plt.setp(ax.get_yticklabels(),fontsize=6); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=180); plt.close(fig); print(f'Test accuracy: {accuracy:.4f} ({support.sum()} images); macro F1: {macro[2]:.4f}; outputs in {out}')
if __name__=='__main__':main()
