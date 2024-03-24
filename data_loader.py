import os
import numpy as np
import cv2
import keras
from skimage.io import imread
from skimage.transform import resize

class CustomGenerator(keras.utils.Sequence) :
  
    def __init__(self, batch_size) :# image_filenames, labels,
        # self.image_filenames = image_filenames
        # self.labels = labels
        self.batch_size = batch_size
        #self.clips_path=self.create_clips_path()

    def __len__(self) :
        return (np.ceil(len(self.clips_path) / float(self.batch_size))).astype(np.int)


    def __getitem__(self, idx) :
        #sequence_size=10
        batch_x = self.clips_path[idx * self.batch_size : (idx+1) * self.batch_size]
        
        clips=np.zeros(shape=(len(batch_x),10, 256, 256, 1))
        for i in range(len(batch_x)):
            #print(i,len(batch_x[i]))
            clip = np.zeros(shape=(len(batch_x[i]), 256, 256, 1))
            for j in range(len(batch_x[i])):
                path =batch_x[i][j]
                img=self.load_npzAsImg(path)
                img=self.pre_process(img)
                clip[j,:,:,:]=img
            # clips.append(np.copy(clip))
            clips[i,:,:,:,:] = np.copy(clip)

        # batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        #return x,y
        return clips,clips

    def load_npzAsImg(self,path):
        nz=np.load(path)
        x=nz['state']
        x = np.transpose(x, (1,2,0))
        return x
    def pre_process(self,img):
        img=resize(img,(256, 256, 1))
        #img/=256.0
        #exit()
        return img
    def create_clips_path(self,rootPath='data/archive (6)/0'):
        clip_sequence_size=10
        dirsList=os.listdir(rootPath)
        dirsList=sorted(dirsList)
        framePathes={}
        for dire in dirsList:
            path=rootPath+'/'+dire
            for root,dirs,files in os.walk(path):
                files=sorted(files)
                files=[path+'/' + x for x in files]
                framePathes[dire]=files

        clips_path=[]
        for k in framePathes.keys():
            for i in range(len(framePathes[k])-clip_sequence_size):
                clips_path.append(framePathes[k][i:i+clip_sequence_size])
        
        self.clips_path=clips_path
        return clips_path

def test():
    dg = CustomGenerator(4)
    dg.create_clips_path()
    for i in range(len(dg)):
        print('i>',i)
        clips,y=dg[i]
        for clip in clips:
            print(clip.shape)
            #exit()
            for f in range(clip.shape[0]):
                frame=clip[f]
                #print(f)
                cv2.imshow('test',frame)
                cv2.waitKey(0)

#test()

# dirPath='/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/archive (6)/NORMAL-TRAIN/ep-0000/ep-0000'

# for root,dirs,files in os.walk(dirPath):
#     files=sorted(files)
#     for f in files:
#         #print(f,os.path.splitext(f)[1])
#         if os.path.splitext(f)[1] == '.npz':
#             nz=np.load(os.path.join(root,f))
#             print(f)
#             print(nz.files)
#             print(nz['action'].shape)
#             print(nz['done'].shape)
#             print(nz['info'].shape)
#             print(nz['state'].shape)
#             print(nz['reward'].shape)
#             print(nz.files)
#             print(nz.files)
#             x=nz['state']
#             x = np.transpose(x, (1,2,0))
#             cv2.imshow('test',x)
#             cv2.waitKey(0)