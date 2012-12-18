from PIL import Image
import colorsys
import argparse
import math
import random
import sys
import numpy as np
import itertools

class Mask:
    
    def __init__(self, dim1, dim2, num=2, min1=1, min2=1, step1=100, step2=20, subdivide=False, vary=False):
        self.dim1 = dim1
        self.dim2 = dim2
        self.min1 = min1
        self.min2 = min2
        self.step1 = step1
        self.step2 = step2
        self.subdivide = subdivide
        self.vary = vary
                
        if not step2:
            step2 = step1
            
        # self.mask = np.empty((dim2,dim1))
        b1 = dim1/step1
        b2 = dim2/step2
        m1 = np.random.randint(num,size=(b2, b1))
        m1 = np.repeat(m1, step1, axis=1)
        self.mask = np.repeat(m1, step2, axis=0)
        # for x in range(0,step1):
        #     for y in range(0, step2):
        #         self.mask[y:dim2:step2, x:dim1:step1] = m1
        # self.mask = self.mask.astype('bool')
        # self.mask = np.expand_dims(self.mask, axis=2)
        
    def get_instance(self, i):
        m = np.equal(self.mask, np.ones_like(self.mask) * i)
        m = m.astype('float')
        return np.expand_dims(m, axis=2)
        
    def include(self):
        return self.mask.astype('float')
        
    def exclude(self):
        return np.logical_not(self.mask).astype('float')

class ColorSpaceConverter:
    
    def __init__(self, bits=255.0):
        self.bits = float(bits)
        
    def rgb_to_hsv(self, t):
        r,g,b = [x/self.bits for x in t]
        return colorsys.rgb_to_hsv(r,g,b)
        
    def hsv_to_rgb(self, t):
        h,s,v = t
        r,g,b = colorsys.hsv_to_rgb(h,s,v)
        return [int(x*self.bits) for x in (r,g,b)]

class Stepper(object):
    
    def __init__(self, xstep, ystep, minx, miny, vary):
        self.xstep = xstep
        self.ystep = ystep if ystep else xstep
        self.vary = vary
        if self.vary:
            self.minx = minx
            self.miny = miny
            
    def next_x_step(self):
        if self.vary:
            step = random.randint(self.minx, self.xstep)
            return step
        else:
            return self.xstep
            
    def next_y_step(self):
        if self.vary:
            step = random.randint(self.miny, self.ystep)
            return step
        else:
            return self.ystep


class Frame(object):
    
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    def intersect(self, p1, p2):
        return (max(self.p1[0], p1[0]), max(self.p1[1], p1[1])), (min(self.p2[0], p2[0]), min(self.p2[1], p2[1]))
        
    def write(self, im, pixels):
        p1,p2 = self.intersect((0,0), im.size)
        img_px = im.load()
        for x in range(p1[0], p2[0]):
            for y in range(p1[1], p2[1]):
                img_px[x,y] = tuple(pixels[(x-p1[0]) * (y-p1[1]), :])
    
    def read(self, im):
        p1,p2 = self.intersect((0,0), im.size)
        img_px = im.load()
        px = [img_px[x,y] for x in range(p1[0], p2[0]) for y in range(p1[1], p2[1])]
        pixels = np.asarray(px)      
        return pixels  
        
    
class PixelBlender(object):
    
    def __init__(self, blended_image, source_images, stepper, xfirst=True, subdivide=False):   
        self.blended_image = blended_image     
        self.width, self.height = blended_image.size
        self.source_images = source_images
        self.stepper = stepper
        self.xfirst = xfirst
        self.subdivide = subdivide
        
    def next_x(self):
        self.current_xstep = stepper.next_x_step()
  
    def next_y(self):
        self.current_ystep = stepper.next_y_step()
        
    def generate_image(self):
        if self.xfirst:
            x=0
            while x < width:
                self.next_x()
                y = 0
                while y < height:
                    self.next_y()
                    self.process_frame(x,y)
                    y += self.current_ystep
                x += self.current_xstep
        else:
            y=0
            while y < height:
                self.next_y()
                x = 0
                while x < width:
                    self.next_x()
                    self.process_frame(x,y)
                    x += self.current_xstep
                y += self.current_ystep
    
    def get_frames(self, x, y):
        frame = Frame((x,y), (x+self.current_xstep,y+self.current_ystep))
        if self.subdivide:
            subx = random.randint(1,self.current_xstep)
            if not self.stepper.vary and self.current_xstep == self.current_ystep:
                suby = subx
            else:
                suby = random.randint(1,self.current_ystep)
            frame_list = []
            for i in range(x,x+self.current_xstep,subx):
                for j in range(y,y+self.current_ystep,suby):
                    subframe = Frame((i,j), (i+subx, j+suby))
                    p1,p2 = frame.intersect(subframe.p1, subframe.p2)
                    frame_list.append(Frame(p1,p2))
            return frame_list
        else:
            return [frame]
                
    def process_frame(self, x, y):
        for frame in self.get_frames(x, y): 
            p = self.get_blended_frame(frame)
            frame.write(self.blended_image, p)
        
    def get_blended_frame(self, frame):
        source_im = random.choice(self.source_images)
        return frame.read(source_im)
        
class ContinuousBlender(PixelBlender):
    
    def get_weights(self, size):
        weights = [0.0] * size
        weights[0] = random.random()
        for i in range(1,len(weights)-1):
            weights[i] = random.uniform(0, weights[i-1])
        weights[-1] = 1.0 - sum(weights[:-1])
        random.shuffle(weights)
        return weights
    
    def get_blended_pixels(self, width, height, weights, pixel_list):
        results = np.zeros((width, height)).astype('float')
        for x in range(0,width):
            for y in range(0,height):
                px = [p.getpixel(x,y) for p in pixel_list]
                weighted_pixels = [(r*w,g*w,b*w) for (r,g,b), w in zip(px, weights)]
                for t1 in weighted_pixels:
                    t2 = results.getpixel(x,y)
                    results.putpixel(x,y, (t1[0] + t2[0], t1[1]+t2[1], t1[1]+t2[2]))
                results.putpixel(x, y, tuple([int(a+0.5) for a in results.getpixel(x,y)]))
        return results
                      
    def get_blended_frame(self, frame):
        weights = self.get_weights(len(self.source_images))
        pixel_list = [frame.read(im) for im in self.source_images]
        return self.get_blended_pixels(pixel_list[0].width, pixel_list[0].height, weights, pixel_list )
 
        
class LuminosityBlender(ContinuousBlender):
            
    def get_blended_pixels(self, width, height, weights, pixel_list):
        results = np.zeros((width, height)).astype('float')
        csv = ColorSpaceConverter()
        max_weight_index = weights.index(max(weights))
        for x in range(0,width):
            for y in range(0,height):
                hsvx = [csv.rgb_to_hsv(p.getpixel(x,y)) for p in pixel_list]
                hmax, smax, vmax = hsvx[max_weight_index] 
                vblend = sum([v*w for (h,s,v), w in zip(hsvx, weights)])
                results.putpixel(x, y, csv.hsv_to_rgb((hmax,smax,vblend)))
        return results    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Randomly blends pixels from source image files.  All the images will be resized to the size of the first one.")
    parser.add_argument("infiles", help="One or more input image files [stdin].", nargs='*', default=sys.stdin, type=argparse.FileType('r'))
    parser.add_argument("--base", help="Base path for output files [../blended/blend].", default="../blended/blend")
    parser.add_argument("--variants", help="Number of variants to generate [1]", type=int, default=1)
    parser.add_argument("--continuous", help="Blend values for each pixel from all images.", action='store_true')
    parser.add_argument("--step", help="Number of pixels in each square[1].", type=int, default=1)
    parser.add_argument("--ystep", help="Number of pixels in y direction[none].  If a value is provided then step becomes the x step size.", type=int)
    parser.add_argument("--vary", help="Vary the step size.", action='store_true')
    parser.add_argument("--minstep", help="Minimum step size (only used if vary is set.)[1]", type=int, default=1)
    parser.add_argument("--minystep", help="Minimum step size in the vertical direction (only used if vary is set.)[1]", type=int, default=1)  
    parser.add_argument("--yfirst", help="Process y direction first", action='store_true')
    parser.add_argument("--luminosity", help="Luminosity blend.", action='store_true')
    parser.add_argument("--subdivide", help="Subdivide squares.", action='store_true')
    
    args = parser.parse_args()
    
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    
    source_images = [Image.open(x) for x in args.infiles]
    width,height = source_images[0].size
    print width, height
    for i in range(1,len(source_images)):
        source_im = source_images[i]
        if source_im.size != (width,height):
            source_images[i] = source_im.resize((width,height), Image.BICUBIC)
            
    source_arrays = [np.asarray(im).astype('float') for im in source_images]
    
    mask = Mask(width, height, num=len(source_images))
    blended_arrays = [reduce(lambda x,y: x+y, [sa * mask.get_instance(i) for (i, sa) in enumerate(source_arrays)]) for j in range(args.variants)]
    for ba in blended_arrays:
        print ba.shape
    for (i, arr) in enumerate(blended_arrays):
        im = Image.fromarray(arr.astype('uint8'))
        im.save(args.base+str(i)+'.png', 'PNG')
    # blended_images = [Image.new('RGB', (width,height)) for x in range(args.variants)]
    # stepper = Stepper(args.step, args.ystep, args.minstep, args.minystep, args.vary)
    # for blended_im in blended_images:
    #     if args.continuous:
    #         if args.luminosity:
    #             b = LuminosityBlender(blended_im, source_images, stepper, not args.yfirst, args.subdivide)
    #         else:
    #             b = ContinuousBlender(blended_im, source_images, stepper, not args.yfirst, args.subdivide)
    #     else:
    #         b = PixelBlender(blended_im, source_images, stepper, not args.yfirst, args.subdivide)
    #         
    #     b.generate_image()
    
    # for i in range(args.variants):
    #     blended_im = blended_images[i]
    #     blended_im.save(args.base+str(i)+'.png', 'PNG')     
    