__author__ = 'jeremy'

from math import hypot, pi, cos, sin
import Image


def hough(im, ntx=460, mry=360):
    "Calculate Hough transform."
    pim = im.load()
    nimx, mimy = im.size
    print('image size:'+str(nimx)+'x'+str(mimy))
    mry = int(mry/2)*2          #Make sure that this is even
    him = Image.new("L", (ntx, mry), 255)
    phim = him.load()

    rmax = hypot(nimx, mimy)
    dr = rmax / (mry/2)
    dth = pi / ntx

    for jx in xrange(nimx):
        print('jx='+str(jx)+' of '+str(nimx))

        for iy in xrange(mimy):
      #      print('iy='+str(iy)+' of '+str(mimy))
            col = pim[jx, iy]
            if col == 255: continue
            for jtx in xrange(ntx):
                th = dth * jtx
                r = jx*cos(th) + iy*sin(th)
                iry = mry/2 + int(r/dr+0.5)
                if(jtx<ntx and iry<mry):    #make sure we a re in bounds - not sure why this is breaking
                    phim[jtx, iry] -= 1
                else:
                    print('jtx:'+str(jtx)+'iry:'+str(iry))
    return him


def test():
    "Test Hough transform with pentagon."
    print('start')
    im = Image.open("testimages/soccer3_600.jpg").convert("L")
    him = hough(im)
    him.save("ho5.bmp")
    print('done')


if __name__ == "__main__": test()
