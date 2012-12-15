from PIL import Image
import colorsys
import argparse
import math
import random
import sys
from itertools import izip

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

class Pixels(object):
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = []
        for x in range(0,width):
            self.pixels.append([(0.0, 0.0, 0.0)] * height)
    
    def putpixel(self, x, y, point):
        self.pixels[x][y] = point
        
    def getpixel(self, x, y):
        return self.pixels[x][y]

class Frame(object):
    
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    def intersect(self, p1, p2):
        return (max(self.p1[0], p1[0]), max(self.p1[1], p1[1])), (min(self.p2[0], p2[0]), min(self.p2[1], p2[1]))
        
    def write(self, im, pixels):
        p1,p2 = self.intersect((0,0), im.size)
        for x in range(p1[0], p2[0]):
            for y in range(p1[1], p2[1]):
                r,g,b = pixels.getpixel(x-p1[0], y-p1[1])
                im.putpixel((x,y), (r,g,b))
    
    def read(self, im):
        p1,p2 = self.intersect((0,0), im.size)
        width = (p2[0] - p1[0]) 
        height = (p2[1]-p1[1])
        pixels = Pixels(width,height)
        for x in range(p1[0], p2[0]):
            for y in range(p1[1], p2[1]):
                r,g,b = im.getpixel((x,y)) 
                pixels.putpixel(x-p1[0], y-p1[1], (r,g,b))  
                
        return pixels  
        

def get_weights(size):
    weights = [0.0] * size
    weights[0] = random.random()
    for i in range(1,len(weights)-1):
        weights[i] = random.uniform(0, weights[i-1])
    weights[-1] = 1.0 - sum(weights[:-1])
    random.shuffle(weights)
    return weights

def choose_one(source_images, x, y):
    source_im = random.choice(source_images)
    return source_im.getpixel((x,y))
    
def choose_blend(source_images, x, y):
    weights = get_weights(len(source_images))
    colors = (im.getpixel((x,y)) for im in source_images)
    weighted_colors = ((ri*w, gi*w, bi*w) for (ri,gi,bi), w in izip(colors, weights))
    r,g,b = (0.0, 0.0, 0.0)
    for ri,gi,bi in weighted_colors:
        r += ri
        g += gi
        b += bi
    
    return [int(x+0.5) for x in (r,g,b)]
    
def get_blended_frame(source_images, frame):
    weights = get_weights(len(source_images))
    pixel_list = [frame.read(im) for im in source_images]
        
    width = pixel_list[0].width
    height = pixel_list[0].height
    results = Pixels(width, height)
    for x in range(0,width):
        for y in range(0,height):
            px = [p.getpixel(x,y) for p in pixel_list]
            weighted_pixels = [(r*w,g*w,b*w) for (r,g,b), w in zip(px, weights)]
            for t1 in weighted_pixels:
                t2 = results.getpixel(x,y)
                results.putpixel(x,y, (t1[0] + t2[0], t1[1]+t2[1], t1[1]+t2[2]))

            results.putpixel(x, y, tuple([int(a+0.5) for a in results.getpixel(x,y)]))
        
    return results
    
def choose_one_frame(source_images, frame):
    source_im = random.choice(source_images)
    return frame.read(source_im)

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
    
    args = parser.parse_args()
    
    source_images = [Image.open(x) for x in args.infiles]
    width,height = source_images[0].size
    for i in range(1,len(source_images)):
        source_im = source_images[i]
        if source_im.size != (width,height):
            source_images[i] = source_im.resize((width,height), Image.BICUBIC)
            
    blended_images = [Image.new('RGB', (width,height)) for x in range(args.variants)]
    stepper = Stepper(args.step, args.ystep, args.minstep, args.minystep, args.vary)
    x=0
    while x < width:
        xstep = stepper.next_x_step()
        y = 0
        while y < height:
            ystep = stepper.next_y_step()
            for blended_im in blended_images:
                if args.continuous:
                    frame =  Frame((x,y), (x+xstep,y+ystep))
                    p = get_blended_frame(source_images, frame)
                    frame.write(blended_im, p)
                    # (r,g,b) = choose_blend(source_images, x, y)
                    # blended_im.putpixel((x,y), (r,g,b))
                else:
                    # (r,g,b) = choose_one(source_images, x, y)
                    # blended_im.putpixel((x,y), (r,g,b))
                    frame =  Frame((x,y), (x+xstep,y+ystep))
                    p = choose_one_frame(source_images, frame)
                    frame.write(blended_im, p)
            y += ystep
        x += xstep
    
    
    for i in range(args.variants):
        blended_im = blended_images[i]
        blended_im.save(args.base+str(i)+'.png', 'PNG')     
    