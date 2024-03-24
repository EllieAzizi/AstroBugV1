import tarfile
import os
import cv2
import numpy as np
from tqdm import tqdm

# path='/home/ghavitan/Pictures/Screenshot from 2023-01-09 09-45-08.png'
# img=cv2.imread(path).astype(np.float32)
# img/=256.0
# img/=256.0
# print(img[..., 0].min())
# print(img[..., 1].max())
# print(img[..., 2].max())
# cv2.imshow('00',img)
# cv2.waitKey(0)
# exit()

# dirPath='/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/Test_data_unzip/CameraClipping'
# for root,dirs,files in os.walk(dirPath):
#     files=sorted(files)
#     for f in files:
#         #print(f,os.path.splitext(f)[1])
#         if os.path.splitext(f)[1] == '.npz':
#             nz=np.load(os.path.join(root,f))
#             print(f)
#             print(nz.files)
#             # print(nz['action'].shape)
#             # print(nz['done'].shape)
#             # print(nz['info'].shape)
#             print(nz['state'].shape)
#             # print(nz['reward'].shape)
#             # print(nz.files)
#             # print(nz.files)
#             x=nz['state']
#             x = np.transpose(x, (1,2,0))
#             cv2.imshow('test',x)
#             cv2.waitKey(0)

def extracting_files():
    # open file
    #file = tarfile.open('/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/archive (6)/NORMAL-TRAIN/ep-0004/ep-0004.tar')
    sourceDirPath = '/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/TEST/GeometryClipping'
    desteDirPath = '/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/Test_data_unzip/GeometryClipping/'
    for root,dirs,files in os.walk(sourceDirPath):
        for f in tqdm(files):
            file = tarfile.open(os.path.join(root,f))
            destPath=desteDirPath+'/'+root.split('/')[-1]
            file.extractall(destPath)
            file.close()
extracting_files()
exit()
# clip_sequence_size=10
# rootPath='/mnt/hard/ghavitan/0_projects/e_h/anomaly/autoencoder/data/Test_data/Test_data_unzip/BoundaryHole'
# dirsList=os.listdir(rootPath)
# dirsList=sorted(dirsList)
# framePathes={}
# for dire in dirsList:
#     path=rootPath+'/'+dire
#     for root,dirs,files in os.walk(path):
#         files=sorted(files)
#         files=[path+'/' + x for x in files]
#         framePathes[dire]=files

# clips=[]
# for k in framePathes.keys():
#     for i in range(len(framePathes[k])-clip_sequence_size):
#         clips.append(framePathes[k][i:clip_sequence_size])
    
# print(clips[0])
# print(len(clips))
# exit()
# print(len(list(framePathes.keys())))