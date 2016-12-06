import os
import argparse
import cv2
import psycopg2 as pg

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--videoId", required=True)
parser.add_argument("--inputDir", required=True)
parser.add_argument("--dbName", required=True)
parser.add_argument("--dbUser", required=True)
parser.add_argument("--dbPassword", required=True)
args = vars(parser.parse_args())
videoId = args['videoId']
inputDir = ['inputDir']
dbName = args['dbName']
dbUser = args['dbUser']
dbPassword = args['dbPassword']

# CONNECT TO DB
conn = pg.connect("dbname=pipedream user=piper password=letm3in")
cur = conn.cursor()

# GET METADATA
cur.execute("SELECT (num_frames,frame_width,frame_height) FROM video_metadata WHERE video_id={}".format(videoId))
(numFrames, frameWidth, frameHeight) = cur.fetchone()

# LOOP OVER FRAMES, GETTING LANDMARKS AND PUPILS, CREATING TRIANGLES, AND DRAWING MARKED IMAGES
for curFrame in range(1,numFrames+1):

    # CREATE IMG PATH
    imgFileName = "{v}.{f}.png".format(v=videoId, f=curFrame)
    imgPath = os.path.join(inputDir, imgFileName)

    # LOAD IMAGE
    bgrImg = cv2.imread(file)
    if bgrImg is None:
        raise Exception("Unable to load image")

    # GET PUPILS AND LANDMARKS
    cur.execute("SELECT * FROM video_data WHERE video_id={v} AND frame_num={f}}".format(v=videoId, f=curFrame))
    videoData = cur.fetchone()

    pupils = zip(videoData[9:13],videoData[9:13])
    print pupils
    landmarks = zip(videoData[13:149],videoData[13:149])
    print landmarks

    # GET DELAUNAY TRIANGLES
    dims = bgrImg.shape
    rect = (0, 0, dims[1], dims[0]) # NOTE: not using width and height from db
    subdiv = cv2.Subdiv2D()
    subdiv.initDelaunay(rect)
    for landmark in landmarks:
        subdiv.insert(landmark)
    triangleList = subdiv.getTriangleList()
    print(triangleList)

    # DRAW PUPILS
    pupilColor = (255, 0, 255)
    for pupil in pupils:
        cv2.circle(bgrImg, pupil, 2, pupilColor, -1)

    # DRAW LANDMARKS
    landmarkColor = (0, 0, 255)
    for landmark in landmarks:
        cv2.circle(bgrImg, landmark, 2, pupilColor, -1)

    # DRAW TRIANGLES
    delaunayColor = (100, 100, 100)
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            cv2.line(bgrImg, pt1, pt2, delaunayColor, 1, cv2.CV_AA, 0)
            cv2.line(bgrImg, pt2, pt3, delaunayColor, 1, cv2.CV_AA, 0)
            cv2.line(bgrImg, pt3, pt1, delaunayColor, 1, cv2.CV_AA, 0)

    # OVERWRITE ORIGINAL IMAGE
    cv2.imwrite(imgPath, bgrImg)

cur.close()
conn.close()