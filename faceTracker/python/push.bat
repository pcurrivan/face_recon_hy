@echo off
pscp -i "C:\Users\pcurr\Documents\aws\aws_putty_key1.ppk" ./triangles.py ubuntu@ec2-52-91-31-149.compute-1.amazonaws.com:/home/ubuntu/
pause