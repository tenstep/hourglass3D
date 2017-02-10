import h5py
import numpy as np
import sys
import h36m

keys = ['index','imgname','center','scale','part_2D','part_3D','istrain']
annot = {k:[] for k in keys}


# Set up index reference for multiperson training
nImage = 0
# Get image filenames
for idx in xrange(len(h36m.imglist)):
    print "\r",idx,
    sys.stdout.flush()

    center,scale,parts_2D,parts_3D = h36m.get_info(idx)
    for f in xrange(len(scale)):
        annot['index'] += [nImage];
        nImage = nImage + 1
        refname = str(h36m.imglist[idx][0])+ '/' + str(5*f-4)+'.jpg'
        imgname = np.zeros(55)
        for i in range(len(refname)): imgname[i] = ord(refname[i])
        annot['imgname'] += [imgname]
        annot['center'] += [center[f,:].tolist()]
        annot['scale']  += [scale[f]]
        annot['part_2D'] += [parts_2D[f,:,:]]
        annot['part_3D'] += [parts_3D[f,:,:]]
        if idx < len(h36m.trainlist):
            annot['istrain'] += [1]
        else:
            annot['istrain'] += [0]

with h5py.File('h36m-annot.h5','w') as f:
    f.attrs['name'] = 'h36m'
    for k in keys:
        f[k] = np.array(annot[k])

print(nImage)