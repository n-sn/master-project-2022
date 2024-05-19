import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import json
import sys
import raisedhandsmodule
import keyboard
import time






#sys.argv
sysargv = 'allposes.json'

with open(sysargv) as f:
    data = json.load(f)

counter = 0
posescounter = 0
for pose in data:
    
    file = pose['filename']

    bbox = pose['bbox']
    testimage = Image.open(file)
    draw = ImageDraw.Draw(testimage)
    draw.rectangle([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]], outline = "red")
    testimage.show()
    time.sleep(2)

    print("please label the selected in the bounding box pose by pressing the r button if the hand is raised, or, in case the hand is not raised, by pressing the n button")

    if keyboard.read_key() == "r":
        pose["raisedhand"] = 1
    elif keyboard.read_key() == "n":
        pose["raisedhand"] = 0
    
    testimage.close()
    posescounter = posescounter + 1
    
    print("noticed " + str(pose["raisedhand"]) + " for " + pose["filename"])
    print(posescounter)
    
print("finished everything, saving")
    
with open('allposes_labeled.json','w') as f:
    json.dump(data, f)
    

print('total files: ', posescounter)
    

