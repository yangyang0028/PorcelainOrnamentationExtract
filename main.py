import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


import numpy as np
from PIL import Image
import glob

import cv2
import math
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    imo.save(d_dir+imidx+'.png')

def Gs(x):
    mu=0
    sigma=30000
    return (1/(math.sqrt(2*math.pi*sigma)))*math.e**(-1.0*(x-mu)*(x-mu)/sigma*2)

def Eu(x1,y1,x2,y2):
    l=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return Gs(l)


def main():


    model_name='u2net'


    image_dir = './test_data/test_images/'
    prediction_dir = './test_data/' + model_name + '_results/'
    model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'

    img_name_list = glob.glob(image_dir + '*')
    print(img_name_list)


    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)


    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor) 

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7
    
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print(i_test,"/",len(test_salobj_dataloader))
        image1 = cv2.imread(img_name_list[i_test])
        ans=image1 
        image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
        ans1=image1
        img_name = img_name_list[i_test].split("/")[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]
        image2 = cv2.imread(prediction_dir+imidx+'.png')
        image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
        image=image2-image1
        m,n =image1.shape
        print("m = ",m,"n = ",n)
        m2=m/2
        n2=n/2
        ans_num=0
        num_num=0
        for i in range(m):
            for j in range(n):
                if image2[i][j]!=0:
                    ans_num+=image[i][j]
                    num_num+=1

        print("threshold   ",(ans_num/num_num+150)/2)

        image3,image=cv2.threshold(image, (ans_num/num_num+150)/2, 255, cv2.THRESH_TOZERO)

        for i in range(m):
            for j in range(n):
                if image[i,j]==0 or image2[i][j]==0:
                    ans[i,j,0]=ans[i,j,1]=ans[i,j,2]=0
                    ans1[i][j]=0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
        ans1 = cv2.morphologyEx(ans1, cv2.MORPH_CLOSE, kernel)
        ans2=ans1.copy()
        num=0
        list_ansx=[]
        list_ansy=[]
        dx = [-1, -2, -2, -1, +1, +2, +2, +1]
        dy = [+2, +1, -1, -2, -2, -1, +1, +2]
        for i in range(m):
            for j in range(n):
                if ans1[i][j]!=0:
                    x=i
                    y=j
                    qx = [x, ]
                    qy = [y, ]
                    cnt=0
                    numm=0
                    numx=[x,]
                    numy=[y,]
                    while cnt >= 0 :
                        px = qx[cnt]
                        py = qy[cnt]
                        numm+=Eu(px,py,m2,n2)
                        qx.pop(cnt)
                        qy.pop(cnt)
                        cnt=cnt-1
                        for i in range(8):
                            x = px + dx[i]
                            y = py + dy[i]
                            if x <= 0 or x >= m or y <= 0 or y >= n or ans1[x][y]==0:
                                continue
                            ans1[x][y]=0;
                            qx.insert(0,x)
                            qy.insert(0,y)
                            numx.append(x)
                            numy.append(y)
                            cnt=cnt+1
                    if numm>num:
                        list_ansx=[]
                        list_ansy=[]
                        list_ansx=numx.copy()
                        list_ansy=numy.copy()
                        num=numm

        for i in range(len(list_ansx)):
            
            ans2[list_ansx[i],list_ansy[i]]=0
        for i in range(m):
            for j in range(n):
                if ans2[i][j]!=0:
                    ans[i][j]=0
        cv2.imwrite(prediction_dir+imidx+'.png', ans)


if __name__ == "__main__":
    main()
