import scipy.io as sio
import scipy.misc
import numpy as np

trainlist = sio.loadmat('trainlist.mat')
testlist = sio.loadmat('testlist.mat')
trainlist = trainlist['trainlist'][0]
testlist = testlist['testlist'][0]
imglist = np.concatenate((trainlist,testlist),axis=0)

# Part info
parts = ['pelvis','rHip','rKnee','rAnkle','lHip','lKnee','lAnkle',
'spine','thorax','jaw','head','lShoulder','lElbow','lWrist','rShoulder','rElbow','rWrist']
nparts = len(parts)

def loadimg(idx,f):
    # Load in image
    return scipy.misc.imread(str(imglist[idx][0])+ '/' + str(f)+'.jpg')


def get_info(idx):
    # Return center of person, and scale factor
    annot = sio.loadmat(str(imglist[idx][0])+'/annot.mat')
    annot = annot['annot'][0][0][5][0]
    frame = len(annot)
    center = np.zeros((frame,2))
    scale  = np.zeros(frame)
    parts_2D = np.zeros((frame,nparts,2))
    parts_3D = np.zeros((frame,nparts,3))
    for i in range(0,frame):
        center[i,:] = annot[i]['ctr'][0]
        scale[i] = annot[i]['scale'][0][0]
        parts_2D[i,:,:] = annot[i]['skel2D'].transpose()
        parts_3D[i,:,:] = annot[i]['skel3D'].transpose()

    return center,scale,parts_2D,parts_3D

