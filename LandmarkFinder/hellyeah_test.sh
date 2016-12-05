#!/bin/bash

cd ~/src/OpenFace/build/bin

./FaceLandmarkImg \
	--inputDir ~/input-images \
	--videoId 2 \
	--dbHost localhost \
	--dbName teamhellyeah \
	--dbUser postgres \
	--dbPassword postgres
