import os
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image
import cv2
import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras_layer_normalization import LayerNormalization
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import tarfile
import random
import shutil
#from data_loader import CustomGenerator

class LoadNpzVideo:
    def __init__(self,batch_size=1):
        self.batch_size=batch_size
        

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
        return clips
    
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
    def create_video_path(self,rootPath):
        self.videoName=rootPath.split('/')[-1]
        self.rootPath=rootPath

        clip_sequence_size=10
        dirsList=os.listdir(rootPath)
        dirsList=sorted(dirsList)
        videoFrames=[]
        for frame in dirsList:
            path=rootPath+'/'+frame
            videoFrames.append(path)
            

        clips_path=[]
        for i in range(len(videoFrames)-clip_sequence_size):
            clips_path.append(videoFrames[i:i+clip_sequence_size])
        self.clips_path=clips_path
        return clips_path

class Inferance:
    def __init__(self,batch_size=1):
        self.model=load_model('model/model.hdf5',custom_objects={'LayerNormalization': LayerNormalization})
        self.clip_sequence_size=10
        self.batch_size=batch_size
        

    def evaluate_video(self,videoDir,saveRoot):
        self.saveRoot=saveRoot
        lv=LoadNpzVideo(self.batch_size)
        lv.create_video_path(videoDir)
        sequences_reconstruction_cost=[]
        for i in range(len(lv)):
            clips=lv[i]
            reconstructed_sequences = self.model.predict(clips,batch_size=self.batch_size)
            print(i,reconstructed_sequences.shape,self.batch_size,clips.shape)
            for j in range(reconstructed_sequences.shape[0]):
                c=np.array([np.linalg.norm(np.subtract(clips[j],reconstructed_sequences[j]))])
                sequences_reconstruction_cost.append(c)
        sequences_reconstruction_cost=np.array(sequences_reconstruction_cost)
        sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
        sr = 1.0 - sa
        self.save_result(lv,sr)
        # # plot the regularity scores
        # plt.plot(sr)
        # plt.ylabel('regularity score Sr(t)')
        # plt.xlabel('frame t')
        # plt.show()
        
        # return sr
    
    def save_result(self,lv_ob,sr):
        videoName=lv_ob.videoName
        savePath=self.saveRoot+videoName
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        out = cv2.VideoWriter(savePath+'/'+videoName+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (256,256))
        count=0
        for i in range(len(lv_ob)):
            clips=lv_ob[i]
            for j in range(clips.shape[0]):
                frame=clips[j][0]
                frame*=256
                frame=np.uint8(frame)
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
                frame=cv2.putText(frame,str(count),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                frame=cv2.putText(frame,'{:.2f}'.format(sr[count][0]),(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                cv2.imshow('r',frame)
                cv2.waitKey(10)
                out.write(frame)
                count+=1
        out.release()
        # plot the regularity scores
        print('sr.shape>',sr.shape)
        plt.plot(sr)
        plt.ylabel('regularity score Sr(t)')
        plt.xlabel('frame t')
        # plt.show()
        plt.savefig(savePath+'/'+videoName+'_fig.png')
        plt.clf()
                # cv2.imshow('r',frame)
                # cv2.waitKey(0)


def extracting_files(sourceDirPath,desteDirPath):
    # sourceDirPath = '/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/TEST/GeometryClipping'
    # desteDirPath = '/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/Test_data_unzip/GeometryClipping/'
    for root,dirs,files in os.walk(sourceDirPath):
        for f in tqdm(files):
            file = tarfile.open(os.path.join(root,f))
            destPath=desteDirPath+'/'+'_'.join(root.split('/')[-2:])
            file.extractall(destPath)
            file.close()
    return destPath
def main():
    # lv=LoadNpzVideo()
    # lv.create_video_path('data/Test_data/Test_data_unzip/BoundaryHole/ep-0002')
    # for i in range(len(lv)):
    #     clips=lv[i]
    #     print(clips.shape)
    inf=Inferance(2)
    dirPath='data/Test_data/Test_data_unzip/GeometryClipping/'
    dirList=os.listdir(dirPath)
    dirList=sorted(dirList)
    for d in dirList:
        videoPath=dirPath+d
        inf.evaluate_video(videoPath)
    # inf.evaluate_video()

def main2():
    num=3
    doneList=['BlackScreen','BoundaryHole','CameraClipping','GeometryClipping']
    inf=Inferance(2)
    sourceDirPath = '/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/TEST/'
    desteDirPath = '/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/Test_data_unzip/'
    categuryList=os.listdir(sourceDirPath)
    categuryList=sorted(categuryList)
    categuryList=list(set(categuryList)-set(doneList))
    # print(categuryList,len(categuryList))
    for d in categuryList:
        videoList=os.listdir(sourceDirPath+d)
        videoList=random.sample(videoList, k=3)
        #print(set(videoList))
        for i in range(len(videoList)):
            videoList[i]=sourceDirPath+d+'/'+videoList[i]
            unzipPath=extracting_files(videoList[i],desteDirPath)
            saveResultPath='result'+'/'+unzipPath.split('/')[-1]+'/'
            inf.evaluate_video(unzipPath,saveResultPath)
            shutil.rmtree(unzipPath)
        # videoPath=dirPath+d
    #     inf.evaluate_video(videoPath)

#main2()

def main3():
    sourceDirPath = '/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/Test_data_unzip/ep-0007'
    inf.evaluate_video(unzipPath,saveResultPath)
        # videoPath=dirPath+d
    #     inf.evaluate_video(videoPath)