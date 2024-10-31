import shutil
import os
import cv2

IN = 'vid-in/'
FRAMES = 'frames/'
OUT = 'vid-out/'
TARGETRES = (64, 64)
NFRAMES = 150
FPS = 30

def extract_frames(vidin, outdirname):
    inpath = os.path.join(IN, vidin)
    if not os.path.exists(inpath):
        print(f'lib.py : extract_frames() :: ERROR ::: The file {inpath} does not exist.')
        return
    if not inpath.endswith('.mp4'):
        print(f'lib.py : extract_frames() :: ERROR ::: The file {inpath} does not have a .mp4 extension')
        return
    outdir = os.path.join(OUT, outdirname)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(inpath)
    framenum = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        framefile = os.path.join(outdir, f'{framenum:04d}.png')
        cv2.imwrite(framefile, frame)
        framenum += 1
    cap.release()
