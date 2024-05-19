import json
import math
import sys
import os


def segment_angle(x1,y1,x2,y2):
	return math.atan2(y2-y1,x2-x1)

def get_keypoint_x(data,n,a):
	return data[n]['keypoints'][3*a]

def get_keypoint_y(data,n,a):
	return data[n]['keypoints'][3*a+1]

def get_angle_elbow_to_wrist(data, n):
	return segment_angle(get_keypoint_x(data,n,7),get_keypoint_y(data,n,7),get_keypoint_x(data,n,9),get_keypoint_y(data,n,9))

def get_angle_elbow_to_wristr(data, n):
	return segment_angle(get_keypoint_x(data,n,8),get_keypoint_y(data,n,8),get_keypoint_x(data,n,10),get_keypoint_y(data,n,10))




def get_left_wrist_x(data,n):
	return get_keypoint_x(data,n,9)

def get_left_wrist_y(data,n):
	return get_keypoint_y(data,n,9)

def get_left_elbow_x(data,n):
	return get_keypoint_x(data,n,7)


def get_left_elbow_y(data,n):
	return get_keypoint_y(data,n,7)


def get_right_wrist_x(data,n):
	return get_keypoint_x(data,n,10)

def get_right_wrist_y(data,n):
	return get_keypoint_y(data,n,10)


def get_right_elbow_x(data,n):
	return get_keypoint_x(data,n,8)

def get_right_elbow_y(data,n):
	return get_keypoint_y(data,n,8)


def get_left_shoulder_x(data,n):
	return get_keypoint_x(data,n,5)

def get_left_shoulder_y(data,n):
	return get_keypoint_y(data,n,5)

def get_right_shoulder_x(data,n):
	return get_keypoint_x(data,n,6)

def get_right_shoulder_y(data,n):
	return get_keypoint_y(data,n,6)


def get_nose_x(data,n):
	return get_keypoint_x(data,n,0)

def get_nose_y(data,n):
	return get_keypoint_y(data,n,0)



#coord system starts from top left
def wrist_above_nose(data,n):
	if get_nose_y(data,n) <= 0:
		return False
	if get_nose_x(data,n) <= 0:
		return False
#nose hast to have higher y coords to be lower than the wrist on the picture
	if get_nose_y(data,n) >= get_left_wrist_y(data,n):
		if get_left_wrist_x(data,n) <=0 :
			return False
		if get_left_wrist_y(data,n) <=0:
			return False
		#print('true, nose: ', get_nose_y(data,n), ' wrist: ', get_left_wrist_y(data,n))
		return True
	if get_nose_y(data,n) >= get_right_wrist_y(data,n):
		if get_right_wrist_x(data,n) <=0 :
			return False
		if get_right_wrist_y(data,n) <=0:
			return False
		return True
	else:
		return False


def wrist_above_shoulders(data,n):
#	if get_nose_y(data,n) <= 0:
#		return False
#	if get_nose_x(data,n) <= 0:
#		return False
#nose hast to have higher y coords to be lower than the wrist on the picture
	if get_right_shoulder_y(data,n) >= get_left_wrist_y(data,n):
		if get_left_wrist_x(data,n) <=0 :
			return False
		if get_left_wrist_y(data,n) <= 0:
			return False
		if get_right_shoulder_y(data,n) <= 0:
			return False
		if get_right_shoulder_x(data,n) <= 0:
			return False
		return True
	if get_right_shoulder_y(data,n) >= get_right_wrist_y(data,n):
		if get_right_shoulder_y(data,n) <= 0:
			return False
		if get_right_shoulder_x(data,n) <= 0:
			return False
		if get_right_wrist_x(data,n) <=0 :
			return False
		if get_right_wrist_y(data,n) <= 0:
			return False
		return True
	if get_left_shoulder_y(data,n) >= get_left_wrist_y(data,n):
		if get_left_wrist_x(data,n) <=0 :
			return False
		if get_left_wrist_y(data,n) <= 0:
			return False
		if get_left_shoulder_x(data,n) <=0 :
			return False
		if get_left_shoulder_y(data,n) <= 0:
			return False
		return True
	if get_left_shoulder_y(data,n)	>= get_right_wrist_y(data,n):
		if get_left_shoulder_x(data,n) <=0 :
			return False
		if get_left_shoulder_y(data,n) <= 0:
			return False
		if get_right_wrist_x(data,n) <=0 :
			return False
		if get_right_wrist_y(data,n) <= 0:
			return False
		return True
	else:
		return False



def hand_pointing_up(data,n):






	if segment_angle(get_left_elbow_x(data,n),get_left_elbow_y(data,n),get_left_wrist_x(data,n),get_left_wrist_y(data,n)) <= -0.785:
		if segment_angle(get_left_elbow_x(data,n),get_left_elbow_y(data,n),get_left_wrist_x(data,n),get_left_wrist_y(data,n)) >= -2.356:
			if get_left_wrist_x(data,n) <=0 :
				return False
			if get_left_wrist_y(data,n) <= 0:
				return False
			if get_left_elbow_x(data,n) <=0 :
				return False
			if get_left_elbow_y(data,n) <= 0:
				return False	
			return True 	
	if segment_angle(get_right_elbow_x(data,n),get_right_elbow_y(data,n),get_right_wrist_x(data,n),get_right_wrist_y(data,n)) <= -0.785:
		if segment_angle(get_right_elbow_x(data,n),get_right_elbow_y(data,n),get_right_wrist_x(data,n),get_right_wrist_y(data,n)) >= -2.356:
			if get_right_wrist_x(data,n) <=0 :
				return False
			if get_right_wrist_y(data,n) <= 0:
				return False
			if get_right_elbow_x(data,n) <=0 :
				return False
			if get_right_elbow_y(data,n) <= 0:
				return False
			return True 	
	else:
		return False



def raisedhandscount(inputfile):
	with open(inputfile) as f:
		data = json.load(f)
	raised_hands_counter = 0
	elementcounter = 0
	for element in data:
		print(get_angle_elbow_to_wrist(data,elementcounter))
		print('wrist above nose ', wrist_above_nose(data,elementcounter))
		print('wrist above shoulders', wrist_above_shoulders(data,elementcounter))
		print('hand pointing up within +- 45deg ', hand_pointing_up(data,elementcounter))
		if wrist_above_nose(data,elementcounter):
			raised_hands_counter += 1
		elif wrist_above_shoulders(data,elementcounter):
			raised_hands_counter += 1
		elif hand_pointing_up(data,elementcounter):
			raised_hands_counter += 1
		elementcounter +=1
	#print('raised hands ', raised_hands_counter)
	#print('non-raised hands', elementcounter-raised_hands_counter)

	return str(raised_hands_counter)



#import raisedhandsmodule


def getlist(inputfile):

	os.system("python3 -m openpifpaf.predict " + inputfile + " --json-output --disable-cuda")
	with open(inputfile + ".predictions.json") as f:
		data = json.load(f)
	for element in data:
		element["filename"] = inputfile
	os.system("rm " + inputfile + ".predictions.json")
	return data

#with open('testoutput.json', 'w') as f:
#	json.dump(data, f)





