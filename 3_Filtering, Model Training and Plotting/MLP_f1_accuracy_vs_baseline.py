import json
import raisedhandsmodule
import math
import sklearn
from sklearn import preprocessing
from sklearn.tree import plot_tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
#from sklearn.model_selection import training_curve


print("starting counting")
with open("allposes_labeled.json") as f:
	data = json.load(f)

raisedhands = 0
nonraisedhands = 0	
for element in data:
        if element["raisedhand"] == 1:
        	raisedhands+= 1
        elif element["raisedhand"] == 0:
        	nonraisedhands+= 1

        	
print(".json has following info: raised hands: " + str(raisedhands) + " and non raised hands: " + str(nonraisedhands))



truenegative=0
falsenegative=0
truepositive=0
falsepositive=0


elementcounter = 0
for element in data:
                raised_hands_counter = 0
                if raisedhandsmodule.wrist_above_nose(data,elementcounter):
               		raised_hands_counter += 1
                elif raisedhandsmodule.wrist_above_shoulders(data,elementcounter):
               		raised_hands_counter += 1
                elif raisedhandsmodule.hand_pointing_up(data,elementcounter):
               		raised_hands_counter += 1
               		
                #print(element["raisedhand"])

                if raised_hands_counter == 0:
                        if element["raisedhand"] == 0:
                                truenegative = truenegative + 1
                        if element["raisedhand"] == 1:
                                falsenegative = falsenegative + 1
                else:
                        if element["raisedhand"] == 1:
                                truepositive = truepositive + 1
                        if element["raisedhand"] == 0:
                                falsepositive = falsepositive + 1
                elementcounter = elementcounter + 1
                #print(raised_hands_counter)



print("truepositives: " + str(truepositive) + " falsepositives: " + str(falsepositive) + " truenegatives " + str(truenegative) + " falsenegatives " + str(falsenegative))


manual_accuracy = ((truepositive + truenegative)) / ( truepositive + truenegative + falsepositive + falsenegative)
manual_precision = (truepositive / (truepositive + falsepositive))
manual_recall = (truepositive / (falsenegative+truepositive))
manual_f1 = (2*(manual_precision*manual_recall)/(manual_precision+manual_recall))
print(manual_accuracy)
print(manual_f1)


#filtering part
#at least one elbow or wrist
#at least one shoulder
#at least one
validPoses = []
validPoseCounter = 0
elementcounter = 0
for element in data:
                atLeastOneWristOrElbow = False
                atLeastOneShoulder = False
                atLeastOneNose = False
                #check at least one wrist
                if ( (raisedhandsmodule.get_left_wrist_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_left_wrist_y(data,elementcounter) >= 0) ):
                        atLeastOneWristOrElbow = True                                                               
                if ( (raisedhandsmodule.get_right_wrist_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_right_wrist_y(data,elementcounter) >= 0) ):
                        atLeastOneWristOrElbow = True
                if ( (raisedhandsmodule.get_right_elbow_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_right_elbow_y(data,elementcounter) >= 0) ):
                        atLeastOneWristOrElbow = True
                if ( (raisedhandsmodule.get_left_elbow_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_left_elbow_y(data,elementcounter) >= 0) ):
                        atLeastOneWristOrElbow = True

                if ( (raisedhandsmodule.get_left_shoulder_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_left_shoulder_y(data,elementcounter) >= 0) ):
                        atLeastOneShoulder = True
                if ( (raisedhandsmodule.get_right_shoulder_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_right_shoulder_y(data,elementcounter) >= 0) ):
                        atLeastOneShoulder = True



                if ( (raisedhandsmodule.get_nose_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_nose_y(data,elementcounter) >= 0) ):
                        atLeastOneNose = True
                        #print(raisedhandsmodule.get_nose_x(data,elementcounter))


                #print(' ', atLeastOneNose, ' ', atLeastOneShoulder, ' ', atLeastOneWristOrElbow)
                if atLeastOneNose and atLeastOneShoulder and atLeastOneWristOrElbow:
                        validPoses = validPoses + [element]
                        validPoseCounter += 1
                        #print(element["raisedhand"], 'valid pose saved')

                elementcounter += 1
#print(validPoses)
#print('valid pose counter: ', validPoseCounter)

elementcounter = 0
for element in validPoses:
        #print(raisedhandsmodule.get_nose_x(validPoses,elementcounter))
        elementcounter += 1
print(elementcounter)

oldData = data
data = validPoses
### calculate normvalue
elementcounter = 0
for element in data:
        if ( (raisedhandsmodule.get_left_shoulder_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_left_shoulder_y(data,elementcounter) >= 0) and (raisedhandsmodule.get_right_shoulder_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_right_shoulder_y(data,elementcounter) >= 0) ):
                element["normvalue"] = ( - raisedhandsmodule.get_nose_y(data,elementcounter) ) + ((raisedhandsmodule.get_left_shoulder_y(data,elementcounter) + raisedhandsmodule.get_right_shoulder_y(data,elementcounter)) / 2)
        elif ( (raisedhandsmodule.get_left_shoulder_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_left_shoulder_y(data,elementcounter) >= 0) ):
                element["normvalue"] = (( - raisedhandsmodule.get_nose_y(data,elementcounter) ) + ((raisedhandsmodule.get_left_shoulder_y(data,elementcounter))))
        elif ( (raisedhandsmodule.get_right_shoulder_x(data,elementcounter) >= 0) and (raisedhandsmodule.get_right_shoulder_y(data,elementcounter) >= 0) ):
                element["normvalue"] = (( - raisedhandsmodule.get_nose_y(data,elementcounter) ) + ((raisedhandsmodule.get_right_shoulder_y(data,elementcounter))))
 
                #print(elementcounter)
                #print(element["normvalue"])
        elementcounter += 1

### normvalue now ok

### calculate manual for filtered dataset



raisedhands = 0
nonraisedhands = 0	
for element in data:
        if element["raisedhand"] == 1:
        	raisedhands+= 1
        elif element["raisedhand"] == 0:
        	nonraisedhands+= 1

        	
print("Filtered dataset has following info: raised hands: " + str(raisedhands) + " and non raised hands: " + str(nonraisedhands))



        
        
truenegative=0
falsenegative=0
truepositive=0
falsepositive=0
                        
elementcounter = 0
for element in data:
                raised_hands_counter = 0
                if raisedhandsmodule.wrist_above_nose(data,elementcounter):
               		raised_hands_counter += 1
                elif raisedhandsmodule.wrist_above_shoulders(data,elementcounter):
               		raised_hands_counter += 1
                elif raisedhandsmodule.hand_pointing_up(data,elementcounter):
               		raised_hands_counter += 1
               		
                #print(element["raisedhand"])

                if raised_hands_counter == 0:
                        if element["raisedhand"] == 0:
                                truenegative = truenegative + 1
                        if element["raisedhand"] == 1:
                                falsenegative = falsenegative + 1
                else:
                        if element["raisedhand"] == 1:
                                truepositive = truepositive + 1
                        if element["raisedhand"] == 0:
                                falsepositive = falsepositive + 1
                #print('filtered ', raised_hands_counter)
                elementcounter = elementcounter + 1



print("truepositives: " + str(truepositive) + " falsepositives: " + str(falsepositive) + " truenegatives " + str(truenegative) + " falsenegatives " + str(falsenegative))


filtered_manual_accuracy = ((truepositive + truenegative)) / ( truepositive + truenegative + falsepositive + falsenegative)
filtered_manual_precision = (truepositive / (truepositive + falsepositive))
filtered_manual_recall = (truepositive / (falsenegative+truepositive))
filtered_manual_f1 = (2*(filtered_manual_precision*filtered_manual_recall)/(filtered_manual_precision+filtered_manual_recall))
print('filtered_manual_accuracy', filtered_manual_accuracy)
print('filtered_manual_f1', filtered_manual_f1)
         		



##EXTRACT FEATURES FROM FILTERED DATASET

dist_features = []
angle_features = []
labels = []

elementcounter = 0
for element in data:
        ns = [raisedhandsmodule.get_nose_x(data,elementcounter), raisedhandsmodule.get_nose_y(data,elementcounter)]
        lw = [raisedhandsmodule.get_left_wrist_x(data,elementcounter) ,raisedhandsmodule.get_left_wrist_y(data,elementcounter)]
        rw = [raisedhandsmodule.get_right_wrist_x(data,elementcounter) ,raisedhandsmodule.get_right_wrist_y(data,elementcounter)]
        le = [raisedhandsmodule.get_left_elbow_x(data,elementcounter) ,raisedhandsmodule.get_left_elbow_y(data,elementcounter)]
        re = [raisedhandsmodule.get_right_elbow_x(data,elementcounter) ,raisedhandsmodule.get_right_elbow_y(data,elementcounter)]
        ls = [raisedhandsmodule.get_left_shoulder_x(data,elementcounter) ,raisedhandsmodule.get_left_shoulder_y(data,elementcounter)]
        rs = [raisedhandsmodule.get_right_wrist_x(data,elementcounter) ,raisedhandsmodule.get_right_wrist_y(data,elementcounter)]
        lh = [raisedhandsmodule.get_keypoint_x(data,elementcounter,11), raisedhandsmodule.get_keypoint_y(data,elementcounter,11)]
        rh = [raisedhandsmodule.get_keypoint_x(data,elementcounter,12), raisedhandsmodule.get_keypoint_y(data,elementcounter,12)]

        keypoints = [ns,lw,rw,le,re,ls,rs,lh,rh]
        #print(keypoints)
        distfeatures = []
        anglefeatures = []
        #print(elementcounter)
        #print('normvalue', element["normvalue"])

        for element1 in keypoints: #for each keypoint
                for element2 in keypoints: #take each keypoint and do
                        if element1[0]>=0 and element1[1]>0 and element2[0]>=0 and element2[1]>=0:
                                distfeatures = distfeatures + [math.dist(element1,element2) / element["normvalue"]]
                                if (raisedhandsmodule.segment_angle(element1[0], element1[1] ,element2[0], element2[1]))!=0:
                                        anglefeatures = anglefeatures + [math.pi + raisedhandsmodule.segment_angle(element1[0], element1[1] ,element2[0], element2[1])]
                                else:
                                        anglefeatures = anglefeatures + [0]
                        else:
                                distfeatures = distfeatures + [0]
                                anglefeatures = anglefeatures + [0]
        #print(distfeatures)
        element["features1"] = distfeatures
        element["features2"] = anglefeatures
        #print(anglefeatures)
        dist_features = dist_features + [distfeatures]
        angle_features = angle_features + [anglefeatures]
        labels = labels + [element["raisedhand"]]
        elementcounter += 1


##now data has dictionaries with features in features1 and features2

##now we normalize and scale stuff

        

min_max_scaler1 = preprocessing.MinMaxScaler()
dist_features_scaled = min_max_scaler1.fit_transform(dist_features)
min_max_scaler2 = preprocessing.MinMaxScaler()
angle_features_scaled = min_max_scaler2.fit_transform(angle_features)

features = np.concatenate((dist_features_scaled,angle_features_scaled), axis=1)

X = features
y = labels

#print(features)

"""
tree = DecisionTreeClassifier()
tree = tree.fit(X,y)

baggedTree = BaggingClassifier()
baggedTree = baggedTree.fit(X,y)


print('Tree ',cross_val_score(DecisionTreeClassifier(), X, y, cv=5, scoring = "f1").mean() )
print('Bagging ',cross_val_score(BaggingClassifier(), X, y, cv=5, scoring = "f1").mean() )
print('GradientBoosting ', cross_val_score(GradientBoostingClassifier(), X, y, cv=5, scoring = "f1").mean() )
print('MLPClassifier ',cross_val_score(MLPClassifier(max_iter=100000), X, y, cv=5, scoring = "f1").mean() )
print('AdaBoostClassifier ',cross_val_score(AdaBoostClassifier(), X, y, cv=5, scoring = "f1").mean() )
print('ExtraTreesClassifier ',cross_val_score(ExtraTreesClassifier(), X, y, cv=5, scoring = "f1").mean() )
print('RandomForest ', cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring = "f1").mean() )
print('GradientBoostingClassifier ', cross_val_score(GradientBoostingClassifier(), X, y, cv=5, scoring = "f1").mean() )
print('HIstGradientBoostingClassifier ',cross_val_score(HistGradientBoostingClassifier(), X, y, cv=5, scoring = "f1").mean() )
#print(cross_val_score(baggedTree, X, y, cv=5, scoring = "f1").mean() )

"""




#dtc = DecisionTreeClassifier()
#dtc = dtc.fit(X,y)

#btc = BaggingClassifier()
#btc = btc.fit(X,y)
"""
gbc = GradientBoostingClassifier()
gbc = gbc.fit(X,y)
"""
mlpc = MLPClassifier(max_iter=100000,activation="relu", solver="sgd",learning_rate_init=0.01,validation_fraction=0.125)
mlpc = mlpc.fit(X,y)











"""

abc = AdaBoostClassifier()
abc = abc.fit(X,y)
"""

#etc = ExtraTreesClassifier()
#etc = etc.fit(X,y)

#rfc = RandomForestClassifier()
#rfc = rfc.fit(X,y)
"""
hgc = HistGradientBoostingClassifier()
hgc = hgc.fit(X,y)

"""
#tree.plot_tree(dtc)



"""
plot_tree(dtc, filled=True)
#plt.title("Decision tree trained on all the iris features")
plt.show()
"""



"""
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=iris.feature_names,  
                      class_names=iris.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
"""
dt = []
bt = []
mlp = []
gb = []
ab = []
et = []
rf = []
hg = []
yx = []
am = []
fm = []




dtA = []
btA = []
mlpA = []
gbA = []
abA = []
etA = []
rfA = []
hgA = []
yxA = []
amA = []
fmA = []




dtF = []
btF = []
mlpF = []
gbF = []
abF = []
etF = []
rfF = []
hgF = []
yxF = []
amF = []
fmF = []


mlpA = []
mlpF =[]

"""
for count in range(4,16):      
        dt.append(cross_val_score(dtc, X, y, cv=count, scoring = "f1").mean() )
        bt.append(cross_val_score(btc, X, y, cv=count, scoring = "f1").mean() )
        gb.append(cross_val_score(gbc, X, y, cv=count, scoring = "f1").mean() )
        mlp.append(cross_val_score(mlpc, X, y, cv=count, scoring = "f1").mean() )
        ab.append(cross_val_score(abc, X, y, cv=count, scoring = "f1").mean() )
        et.append(cross_val_score(etc, X, y, cv=count, scoring = "f1").mean() )
        rf.append(cross_val_score(rfc, X, y, cv=count, scoring = "f1").mean() )
        hg.append(cross_val_score(hgc, X, y, cv=count, scoring = "f1").mean())
        yx.append(count)
        am.append(manual_f1)
        fm.append(filtered_manual_f1)





#for count in range(4,16):      
#        dt.append(cross_val_score(DecisionTreeClassifier(), X, y, cv=count, scoring = "f1").mean() )
#        bt.append(cross_val_score(BaggingClassifier(), X, y, cv=count, scoring = "f1").mean() )
#        gb.append(cross_val_score(GradientBoostingClassifier(), X, y, cv=count, scoring = "f1").mean() )
#        mlp.append(cross_val_score(MLPClassifier(max_iter=100000), X, y, cv=count, scoring = "f1").mean() )
#        ab.append(cross_val_score(AdaBoostClassifier(), X, y, cv=count, scoring = "f1").mean() )
#        et.append(cross_val_score(ExtraTreesClassifier(), X, y, cv=count, scoring = "f1").mean() )
#        rf.append(cross_val_score(RandomForestClassifier(), X, y, cv=count, scoring = "f1").mean() )
#        hg.append(cross_val_score(HistGradientBoostingClassifier(), X, y, cv=count, scoring = "f1").mean())
#        yx.append(count)

#print(dt)
#print(yx)


plt.plot(yx,dt, label="DecisionTree")
plt.plot(yx,bt,label="BaggedTrees")
plt.plot(yx,mlp,label="MultiLayerPerceptron")
plt.plot(yx,gb,label="GradientBoosting")
plt.plot(yx,ab,label="AdaBoost")
plt.plot(yx,et,label="ExtraTrees")
plt.plot(yx,rf,label="RandomForest")
plt.plot(yx,hg,label="HistGradientBoosting")
plt.legend()

plt.show()

"""





"""




for count in range(4,20):      
        dt.append(cross_val_score(dtc, X, y, cv=count, scoring = "accuracy").mean() )
        bt.append(cross_val_score(btc, X, y, cv=count, scoring = "accuracy").mean() )
        gb.append(cross_val_score(gbc, X, y, cv=count, scoring = "accuracy").mean() )
        mlp.append(cross_val_score(mlpc, X, y, cv=count, scoring = "accuracy").mean() )
        ab.append(cross_val_score(abc, X, y, cv=count, scoring = "accuracy").mean() )
        et.append(cross_val_score(etc, X, y, cv=count, scoring = "accuracy").mean() )
        rf.append(cross_val_score(rfc, X, y, cv=count, scoring = "accuracy").mean() )
        hg.append(cross_val_score(hgc, X, y, cv=count, scoring = "accuracy").mean())
        yx.append(count)
        am.append(manual_accuracy)
        fm.append(filtered_manual_accuracy)





plt.plot(yx,dt, label="DecisionTree")
plt.plot(yx,bt,label="BaggedTrees")
plt.plot(yx,mlp,label="MultiLayerPerceptron")
plt.plot(yx,gb,label="GradientBoosting")
plt.plot(yx,ab,label="AdaBoost")
plt.plot(yx,et,label="ExtraTrees")
plt.plot(yx,rf,label="RandomForest")
plt.plot(yx,hg,label="HistGradientBoosting")
plt.plot(yx,am,label="accuracy nonfiltered")
plt.plot(yx,fm,label="accuracy filtered")
plt.legend()

plt.show()


"""
"""


### ensemble tree plot


for count in range(5,20):      


        
        dtA.append(cross_val_score(dtc, X, y, cv=count, scoring = "accuracy").mean() )
        dtF.append(cross_val_score(dtc, X, y, cv=count, scoring = "f1").mean() )

        btA.append(cross_val_score(btc, X, y, cv=count, scoring = "accuracy").mean() )
        btF.append(cross_val_score(btc, X, y, cv=count, scoring = "f1").mean() )

        
        #etA.append(cross_val_score(etc, X, y, cv=count, scoring = "accuracy").mean() )
        #etF.append(cross_val_score(etc, X, y, cv=count, scoring = "f1").mean() )


        rfA.append(cross_val_score(rfc, X, y, cv=count, scoring = "accuracy").mean() )
        rfF.append(cross_val_score(rfc, X, y, cv=count, scoring = "f1").mean() )


        #mlpA.append(cross_val_score(mlpc, X, y, cv=count, scoring = "accuracy").mean() )
        #mlpF.append(cross_val_score(bt, X, y, cv=count, scoring = "f1").mean() )

        yx.append(count)
        am.append(filtered_manual_accuracy)
        fm.append(filtered_manual_f1)



fig, axs = plt.subplots(2)
fig.suptitle('Comparative performance of machine learning algorithms based on Decision Trees, with the manual algorithm as baseline')
fig.set_size_inches(10, 7)
axs[0].plot(yx, dtA, label = "Decision Tree Accuracy Score")
axs[1].plot(yx, dtF, label = "Decision Tree F1 Score")
axs[0].plot(yx, am, label = "Baseline (Heuristic) Accuracy Score")
axs[1].plot(yx, fm, label = "Baseline (Heuristic) F1 Score")
axs[0].plot(yx, rfA, label = "Random Forest Accuracy Score", linestyle = "-.")
axs[1].plot(yx, rfF, label = "Random Forest F1 Score", linestyle = "-.")



axs[0].plot(yx, btA, label = "Bagged Tree Accuracy Score")
axs[1].plot(yx, btF, label = "Bagged Tree F1 Score")
axs[0].set_ylabel('Accuracy Score')
axs[1].set_ylabel('F1 Score')


axs[0].set_xlabel('Cross-validation Folds')
axs[1].set_xlabel('Cross-validation Folds')

axs[0].legend(loc=1)
axs[1].legend(loc=1)
plt.show()

"""

###ensemble tree plot end


"""
plt.plot(yx,dt, label="DecisionTree")
plt.plot(yx,bt,label="BaggedTrees")


plt.plot(yx,mlpA,label="MultiLayerPerceptronAccuracy")
#plt.plot(yx,mlpF,label="MultiLayerPerceptronF1")

plt.plot(yx,am,label="accuracy manual")
#plt.plot(yx,fm,label="f1 manual")
plt.legend()

plt.show()
"""







### simple decision tree plot
"""

for count in range(5,20):      


        
        dtA.append(cross_val_score(dtc, X, y, cv=count, scoring = "accuracy").mean() )
        dtF.append(cross_val_score(dtc, X, y, cv=count, scoring = "f1").mean() )
        #bt.append(cross_val_score(btc, X, y, cv=count, scoring = "accuracy").mean() )

        #mlpA.append(cross_val_score(mlpc, X, y, cv=count, scoring = "accuracy").mean() )
        #mlpF.append(cross_val_score(bt, X, y, cv=count, scoring = "f1").mean() )

        yx.append(count)
        am.append(filtered_manual_accuracy)
        fm.append(filtered_manual_f1)



fig, axs = plt.subplots(2)
fig.suptitle('Decision Tree performance, compared to baseline')
fig.set_size_inches(10, 7)
axs[0].plot(yx, dtA, label = "Decision Tree Accuracy Score")
axs[1].plot(yx, dtF, label = "Decision Tree F1 Score")
axs[0].plot(yx, am, label = "Baseline (Heuristic) Accuracy Score")
axs[1].plot(yx, fm, label = "Baseline (Heuristic) F1 Score")
axs[0].set_ylabel('Accuracy Score')
axs[1].set_ylabel('F1 Score')


axs[0].set_xlabel('Cross-validation Folds')
axs[1].set_xlabel('Cross-validation Folds')

axs[0].legend(loc=1)
axs[1].legend(loc=1)
plt.show()
"""


###simple decision tree plot end




### MLP  plot


for count in range(5,20):      


        
        #dtA.append(cross_val_score(dtc, X, y, cv=count, scoring = "accuracy").mean() )
        #dtF.append(cross_val_score(dtc, X, y, cv=count, scoring = "f1").mean() )
        #bt.append(cross_val_score(btc, X, y, cv=count, scoring = "accuracy").mean() )

        mlpA.append(cross_val_score(mlpc, X, y, cv=count, scoring = "accuracy").mean() )
        mlpF.append(cross_val_score(mlpc, X, y, cv=count, scoring = "f1").mean() )

        yx.append(count)
        am.append(filtered_manual_accuracy)
        fm.append(filtered_manual_f1)



fig, axs = plt.subplots(2)
fig.suptitle('Neural Network performance, compared to baseline')
fig.set_size_inches(10, 7)
axs[0].plot(yx, mlpA, label = "Multilayer Perceptron Accuracy Score")
axs[1].plot(yx, mlpF, label = "Multilayer Perceptron F1 Score")
axs[0].plot(yx, am, label = "Baseline (Heuristic) Accuracy Score")
axs[1].plot(yx, fm, label = "Baseline (Heuristic) F1 Score")
axs[0].set_ylabel('Accuracy Score')
axs[1].set_ylabel('F1 Score')


axs[0].set_xlabel('Cross-validation Folds')
axs[1].set_xlabel('Cross-validation Folds')

axs[0].legend(loc=1)
axs[1].legend(loc=1)
plt.show()



### MLP plot end


        
        


#bar chart comparing accuracy and f1 scores for non-filtered and filtered datasets
'''






labels = ['F1 Score', 'Accuracy']
original_dataset = [manual_f1, manual_accuracy]
filtered_dataset = [filtered_manual_f1, filtered_manual_accuracy]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, original_dataset, width, label='Original Dataset')
rects2 = ax.bar(x + width/2, filtered_dataset, width, label='Filtered Dataset')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score')
ax.set_title('Accuracy and F1 Scores for the Original Dataset and for the Filtered Dataset')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.set_ylim(0, 1)


fig.tight_layout()

plt.show()

'''
###bar chart end

