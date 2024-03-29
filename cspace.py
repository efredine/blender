import Image
import numpy as np

def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    hsv=np.empty_like(rgb)
    hsv[...,3:]=rgb[...,3:]
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    maxc = np.max(rgb[...,:2],axis=-1)
    minc = np.min(rgb[...,:2],axis=-1)
    maxminc = maxc-minc + 0.000001  
    hsv[...,2] = maxc   
    hsv[...,1] = maxminc / (maxc + 0.000001)
    rc = (maxc-r) / maxminc #(maxc-minc)
    gc = (maxc-g) / maxminc #(maxc-minc)
    bc = (maxc-b) / maxminc #(maxc-minc)
    hsv[...,0] = np.select([r==maxc,g==maxc],[bc-gc,2.0+rc-bc],default=4.0+gc-rc)
    hsv[...,0] = (hsv[...,0]/6.0) % 1.0
    idx=(minc == maxc)
    hsv[...,0][idx]=0.0
    hsv[...,1][idx]=0.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    rgb=np.empty_like(hsv)
    rgb[...,3:]=hsv[...,3:]    
    h,s,v=hsv[...,0],hsv[...,1],hsv[...,2]   
    i = (h*6.0).astype('uint8')
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    conditions=[s==0.0,i==1,i==2,i==3,i==4,i==5]
    rgb[...,0]=np.select(conditions,[v,q,p,p,t,v],default=v)
    rgb[...,1]=np.select(conditions,[v,v,v,q,p,p],default=t)
    rgb[...,2]=np.select(conditions,[v,p,t,v,v,q],default=p) 
    return rgb

def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb


if __name__=='__main__':
    img = Image.open('test.png').convert('RGBA')
    arr=np.array(np.asarray(img).astype('float'))
    
    green_hue=(180-78)/360.0
    red_hue=(180-180)/360.0

    new_img = Image.fromarray(shift_hue(arr,red_hue).astype('uint8'),'RGBA')
    new_img.save('test_red.png')


    new_img = Image.fromarray(shift_hue(arr,green_hue).astype('uint8'),'RGBA')
    new_img.save('test_green.png')