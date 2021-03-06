#!/usr/bin/env python

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
print "Parsing arguments..."
parser = argparse.ArgumentParser()
parser.add_argument("--videoId", required=True)
parser.add_argument("--inputDir", required=True, help="Path to directory containing image files for frames of this video")
parser.add_argument("--dbName", required=True)
parser.add_argument("--dbUser", required=True)
parser.add_argument("--dbPassword", required=True)
args = vars(parser.parse_args())
videoId = args['videoId']
inputDir = args['inputDir']
dbName = args['dbName']
dbUser = args['dbUser']
dbPassword = args['dbPassword']

# CONNECT TO DB
print "Connecting to DB..."
try:
    conn = pg.connect("dbname={n} user={u} password={p}".format(n=dbName, u=dbUser, p=dbPassword))
except Exception as e:
    print "Error connecting to db", e
    exit(1)

cur = conn.cursor()

# GET METADATA
selectString = "SELECT num_frames,frame_width,frame_height FROM video_metadata WHERE video_id={}".format(videoId)
print "SELECT command for metadata: ", selectString
try:
    cur.execute("SELECT num_frames,frame_width,frame_height FROM video_metadata WHERE video_id={}".format(videoId))
except Exception as e:
    print "Error selecting from metadata table", e
    exit(1)

metadata = cur.fetchone()
if metadata is None: # make sure that worked
    print "metadata is none!"
    exit(1)
print "metadata: ", metadata

(numFrames, frameWidth, frameHeight) = metadata
print "{} frames available for this video".format(numFrames)
print "frame width:",frameWidth
print "frame height:",frameHeight

# LOOP OVER FRAMES, GETTING LANDMARKS AND PUPILS, CREATING TRIANGLES, AND DRAWING MARKED IMAGES
for curFrame in range(1,numFrames+1):

    print "Processing frame ", curFrame

    # CHECK THAT ROW EXISTS AND GET PUPILS AND LANDMARKS
    print "SELECT * FROM video_data WHERE video_id={v} AND frame_num={f}".format(v=videoId, f=curFrame)
    try:
        cur.execute("SELECT * FROM video_data WHERE video_id={v} AND frame_num={f}".format(v=videoId, f=curFrame))
    except Exception as e:
        print "Error selecting from video_data table", e
        exit(1)

    videoData = cur.fetchone()
    if videoData is None:  # make sure that worked
        print "videoData is none! skipping this frame!"
        continue

    print videoData
    print "Length of video data row from select statement: {}".format(len(videoData))

    pupils = videoData[9:13]
    pupils = map(int, pupils)
    pupils = zip(pupils[::2],pupils[1::2])
    print "Pupils:", pupils
    landmarks = videoData[13:149]
    landmarks = map(int, landmarks)
    landmarks = zip(landmarks[::2],landmarks[1::2])
    print "Landmarks:", landmarks

    # CREATE IMG PATH
    imgFileName = "{v}.{f}.png".format(v=videoId, f=curFrame)
    imgPath = os.path.join(inputDir, imgFileName)
    print imgPath

    # LOAD IMAGE
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image")

    # GET DELAUNAY TRIANGLES
    dims = bgrImg.shape
    rect = (0, 0, frameHeight, frameWidth)
    print "Bounding rect:",rect
    subdiv = cv2.Subdiv2D()
    subdiv.initDelaunay(rect)
    for landmark in landmarks:
        if rect_contains(rect,landmark):
            subdiv.insert(landmark)
        else:
            print "Landmark out of bounds:", landmark
    triangleList = subdiv.getTriangleList()
    print(triangleList)

    # DRAW PUPILS
    pupilColor = (255, 0, 255)
    for pupil in pupils:
        cv2.circle(bgrImg, pupil, 2, pupilColor, -1)

    # DRAW LANDMARKS
    landmarkColor = (0, 0, 255)
    for landmark in landmarks:
        if (rect_contains(rect, landmark)):
            cv2.circle(bgrImg, landmark, 2, pupilColor, -1)

    # DRAW TRIANGLES
    delaunayColor = (100, 100, 100)
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            cv2.line(bgrImg, pt1, pt2, delaunayColor, 1, 8, 0)
            cv2.line(bgrImg, pt2, pt3, delaunayColor, 1, 8, 0)
            cv2.line(bgrImg, pt3, pt1, delaunayColor, 1, 8, 0)
        else:
            print "Triangle points out of bounds!",(pt1,pt2,pt3)

    # OVERWRITE ORIGINAL IMAGE
    cv2.imwrite(imgPath, bgrImg)

cur.close()
conn.close()
print "script finished!"
