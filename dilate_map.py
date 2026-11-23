import argparse
import numpy as np


def read_pgm(path):
    with open(path,'rb') as f:
        assert f.readline().strip()==b'P5'
        line=f.readline()
        while line.startswith(b'#'): #skip comment lines
            line=f.readline()
        w,h=map(int,line.split())
        f.readline() #max value, always 255
        return np.frombuffer(f.read(),np.uint8).reshape(h,w)


def dilate(img, px):
    h,w=img.shape
    occ=np.pad(img<10,px) #black pixels, padded so nothing wraps around the edges
    out=np.zeros((h,w),bool)
    for dy in range(-px,px+1):
        for dx in range(-px,px+1):
            if dx*dx+dy*dy<=px*px: #round brush
                out|=occ[px+dy:px+dy+h,px+dx:px+dx+w]
    res=img.copy()
    res[out]=0
    return res


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="grow the walls of a map, e.g. python3 dilate_map.py complete_maze_2 complete_maze_3 --px 2")
    parser.add_argument("input", help="map name without extension")
    parser.add_argument("output", help="new map name without extension")
    parser.add_argument("--px", type=int, default=2, help="pixels to grow by (0.05 m each at the usual resolution)")
    args=parser.parse_args()

    img=read_pgm(args.input+".pgm")
    new=dilate(img,args.px)
    h,w=new.shape
    with open(args.output+".pgm",'wb') as f:
        f.write(f"P5\n{w} {h}\n255\n".encode()+new.tobytes())

    lines=open(args.input+".yaml").read().splitlines()
    lines=[f"image: {args.output}.pgm" if l.startswith("image:") else l for l in lines] #point the yaml at the new pgm
    open(args.output+".yaml",'w').write("\n".join(lines)+"\n")

    print(f"{args.output}.pgm/.yaml written, black px {(img<10).sum()} -> {(new<10).sum()}")
