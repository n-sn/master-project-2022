import raisedhandsmodule
import openpifpaf
import sys
import os
from PIL import Image, ImageFont, ImageDraw
import glob









#filename = sys.argv[1]


#print(filename)









#for file in glob.glob("*"):
def createfile(filename):	
	raisednumber = 0
	os.system("python3 -m openpifpaf.predict " + filename + " --json-output --image-output --disable-cuda")
	raisednumber = raisedhandsmodule.raisedhandscount( filename + '.predictions.json')
	os.system("rm " + filename + ".predictions.json" )
	testimage = Image.open(filename + '.predictions.jpeg')
	title_font = ImageFont.truetype('JetBrainsMono-Medium.ttf', 100)
	title_text = raisednumber
	image_editable = ImageDraw.Draw(testimage)
	image_editable.text((15,15), title_text, (0, 0, 255), font=title_font)
	testimage.save(filename+'counted.jpeg')
	os.system("rm " + filename + ".predictions.jpeg")

#raisedhandsmodule.raisedhandscount(sys.argv[1])




directory = sys.argv[1]



for filenm in os.listdir(directory):
    f = os.path.join(directory, filenm)
    # checking if it is a file
    if os.path.isfile(f):
        createfile(f)

