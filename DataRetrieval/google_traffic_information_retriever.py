# -*- coding: utf-8 -*-
import requests
import time
import random

LARSAPIKEY_1 = "AIzaSyAuWAW34Q4r4_in7Opl6ry1L2G92M8l0tU"
LARSAPIKEY_2 = "AIzaSyAV4paleeEavURgfyFziQw_7RWkNi2mviw"


def cleanCalculateTravelTimeMatrixFromCoordVector(coordVector, transportType, apikey, writeToFile):
	numberOfCoordinates = len(coordVector)
	secondMatrix = [[0 for i in range(numberOfCoordinates)] for j in range(numberOfCoordinates)]
	numberOfQueries = 0
	totalQueries = 0
	for i in range(numberOfCoordinates):
		origin = makeStringListFromCoordinateVector([coordVector[i]])
		for j in range(numberOfCoordinates):
			if(numberOfQueries > 99):
				time.sleep(1)
				numberOfQueries = 0
			destination = makeStringListFromCoordinateVector([coordVector[j]])
			url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=" + origin + "&destinations=" + destination + "&key=" + apikey + "&mode=" + transportType
			data = requests.get(url).json()
			#print(data)
			if ("error_message" in data.keys()):
				print(data["error_message"])
				return secondMatrix
			# retrieve travel time in seconds from json object
			seconds = data["rows"][0]["elements"][0]["duration"]["value"]
			# add travel time to matrix
			secondMatrix[i][j] = seconds
			numberOfQueries += 1
			totalQueries += 1
			#print(totalQueries)
	if(writeToFile):
		writeMatrixToFile(secondMatrix, transportType)
		print(secondMatrix)

	return secondMatrix

def writeMatrixToFile(matrix, transportType):
    if transportType == "":
        transportType = "car"
    fil = open("travelTimes_" + transportType + ".txt", "w")
    writeString = ""
    for row in matrix:
        for elem in row:
            writeString += str(elem) + " "
        writeString = writeString[:-1] + "\n"
    fil.write(writeString[:-1])
    fil.close()


def makeStringListFromCoordinateVector(coordVector):
    ret = ""
    for coor in coordVector:
        ret += str(coor[0]) + "," + str(coor[1]) + "|"
    ret = ret[:-1]
    return ret


def run(coordVector, transportType, writeToFile):
    apikey = LARSAPIKEY_1
    #print("number of coordinates", len(coordVector))
    return cleanCalculateTravelTimeMatrixFromCoordVector(coordVector, transportType, apikey, writeToFile)

#cords = [(59.958428687499996, 10.647886300000001), (59.958428687499996, 10.662206900000001), (59.958428687499996, 10.6765275), (59.958428687499996, 10.690848100000002), (59.958428687499996, 10.705168700000002), (59.958428687499996, 10.719489300000001), (59.958428687499996, 10.7338099), (59.958428687499996, 10.7481305), (59.958428687499996, 10.762451100000002), (59.958428687499996, 10.776771700000001), (59.958428687499996, 10.7910923), (59.9512480625, 10.647886300000001), (59.9512480625, 10.662206900000001), (59.9512480625, 10.6765275), (59.9512480625, 10.690848100000002), (59.9512480625, 10.705168700000002), (59.9512480625, 10.719489300000001), (59.9512480625, 10.7338099), (59.9512480625, 10.7481305), (59.9512480625, 10.762451100000002), (59.9512480625, 10.776771700000001), (59.9512480625, 10.7910923), (59.9512480625, 10.8197335), (59.9512480625, 10.8483747), (59.944067437499996, 10.647886300000001), (59.944067437499996, 10.662206900000001), (59.944067437499996, 10.6765275), (59.944067437499996, 10.690848100000002), (59.944067437499996, 10.705168700000002), (59.944067437499996, 10.719489300000001), (59.944067437499996, 10.7338099), (59.944067437499996, 10.7481305), (59.944067437499996, 10.762451100000002), (59.944067437499996, 10.776771700000001), (59.944067437499996, 10.7910923), (59.944067437499996, 10.8054129), (59.944067437499996, 10.8197335), (59.944067437499996, 10.834054100000001), (59.944067437499996, 10.8483747), (59.93688681249999, 10.647886300000001), (59.93688681249999, 10.662206900000001), (59.93688681249999, 10.6765275), (59.93688681249999, 10.690848100000002), (59.93688681249999, 10.705168700000002), (59.93688681249999, 10.719489300000001), (59.93688681249999, 10.7338099), (59.93688681249999, 10.7481305), (59.93688681249999, 10.762451100000002), (59.93688681249999, 10.776771700000001), (59.93688681249999, 10.7910923), (59.93688681249999, 10.8054129), (59.93688681249999, 10.8197335), (59.93688681249999, 10.834054100000001), (59.93688681249999, 10.8483747), (59.929706187499995, 10.647886300000001), (59.929706187499995, 10.662206900000001), (59.929706187499995, 10.6765275), (59.929706187499995, 10.690848100000002), (59.929706187499995, 10.705168700000002), (59.929706187499995, 10.719489300000001), (59.929706187499995, 10.7338099), (59.929706187499995, 10.7481305), (59.929706187499995, 10.762451100000002), (59.929706187499995, 10.776771700000001), (59.929706187499995, 10.7910923), (59.929706187499995, 10.8054129), (59.929706187499995, 10.8197335), (59.929706187499995, 10.834054100000001), (59.929706187499995, 10.8483747), (59.9225255625, 10.647886300000001), (59.9225255625, 10.662206900000001), (59.9225255625, 10.6765275), (59.9225255625, 10.690848100000002), (59.9225255625, 10.705168700000002), (59.9225255625, 10.719489300000001), (59.9225255625, 10.7338099), (59.9225255625, 10.7481305), (59.9225255625, 10.762451100000002), (59.9225255625, 10.776771700000001), (59.9225255625, 10.7910923), (59.9225255625, 10.8054129), (59.9225255625, 10.8197335), (59.9225255625, 10.834054100000001), (59.9225255625, 10.8483747), (59.915344937499995, 10.647886300000001), (59.915344937499995, 10.6765275), (59.915344937499995, 10.690848100000002), (59.915344937499995, 10.705168700000002), (59.915344937499995, 10.719489300000001), (59.915344937499995, 10.7338099), (59.915344937499995, 10.7481305), (59.915344937499995, 10.762451100000002), (59.915344937499995, 10.776771700000001), (59.915344937499995, 10.7910923), (59.915344937499995, 10.8054129), (59.915344937499995, 10.8197335), (59.915344937499995, 10.834054100000001), (59.915344937499995, 10.8483747), (59.90816431249999, 10.6765275), (59.90816431249999, 10.690848100000002), (59.90816431249999, 10.719489300000001), (59.90816431249999, 10.7338099), (59.90816431249999, 10.7481305), (59.90816431249999, 10.762451100000002), (59.90816431249999, 10.776771700000001), (59.90816431249999, 10.7910923), (59.90816431249999, 10.8054129), (59.90816431249999, 10.8197335), (59.90816431249999, 10.834054100000001), (59.90816431249999, 10.8483747)]
#cords = [(59.958428687499996, 10.647886300000001), (59.958428687499996, 10.662206900000001),(59.958428687499996, 10.6765275)]


#travel_times = run(coordVector=cords, transportType="driving", writeToFile=False)
#s = 0
#div = len(travel_times)*(len(travel_times[0])-1)

#for l in travel_times:
#	s+=sum(l)

#print(s/(div*60))
#result = 13.09744787322769



#print(run(coordVector=cords, transportType="bicycling", writeToFile=False))
#print(run(coordVector=cords, transportType="transit", writeToFile=False))


#def get_coordVector(nNodes):
#	try:
#		f = open('Coordinates/cordFile.txt', "r")
#		coordVector = []
#		for line in f:
#			row = tuple(float(cor) for cor in line.strip().split(','))
#			coordVector.append(row)
#		coordVector = random.sample(coordVector, nNodes)
#		print(coordVector)
#		return coordVector
#	except IOError:
#		return 0
#get_coordVector(6)

#coordVector = [(59.95245396666667, 10.686117766666667), (59.95245396666667, 10.6952993), (59.95245396666667, 10.704480833333333), (59.95245396666667, 10.713662366666668), (59.95245396666667, 10.7228439), (59.95245396666667, 10.732025433333334), (59.95245396666667, 10.741206966666667), (59.95245396666667, 10.7503885), (59.95245396666667, 10.759570033333334), (59.95245396666667, 10.768751566666667), (59.95245396666667, 10.7779331), (59.95245396666667, 10.787114633333333), (59.95245396666667, 10.796296166666666), (59.95245396666667, 10.8054777), (59.95245396666667, 10.814659233333334), (59.9491519, 10.686117766666667), (59.9491519, 10.6952993), (59.9491519, 10.704480833333333), (59.9491519, 10.713662366666668), (59.9491519, 10.7228439), (59.9491519, 10.732025433333334), (59.9491519, 10.741206966666667), (59.9491519, 10.7503885), (59.9491519, 10.759570033333334), (59.9491519, 10.768751566666667), (59.9491519, 10.7779331), (59.9491519, 10.787114633333333), (59.9491519, 10.796296166666666), (59.9491519, 10.8054777), (59.9491519, 10.814659233333334), (59.945849833333334, 10.686117766666667), (59.945849833333334, 10.6952993), (59.945849833333334, 10.704480833333333), (59.945849833333334, 10.713662366666668), (59.945849833333334, 10.7228439), (59.945849833333334, 10.732025433333334), (59.945849833333334, 10.741206966666667), (59.945849833333334, 10.7503885), (59.945849833333334, 10.759570033333334), (59.945849833333334, 10.768751566666667), (59.945849833333334, 10.7779331), (59.945849833333334, 10.787114633333333), (59.945849833333334, 10.796296166666666), (59.945849833333334, 10.8054777), (59.945849833333334, 10.814659233333334), (59.942547766666664, 10.686117766666667), (59.942547766666664, 10.6952993), (59.942547766666664, 10.704480833333333), (59.942547766666664, 10.713662366666668), (59.942547766666664, 10.7228439), (59.942547766666664, 10.732025433333334), (59.942547766666664, 10.741206966666667), (59.942547766666664, 10.7503885), (59.942547766666664, 10.759570033333334), (59.942547766666664, 10.768751566666667), (59.942547766666664, 10.7779331), (59.942547766666664, 10.787114633333333), (59.942547766666664, 10.796296166666666), (59.942547766666664, 10.8054777), (59.942547766666664, 10.814659233333334), (59.9392457, 10.686117766666667), (59.9392457, 10.6952993), (59.9392457, 10.704480833333333), (59.9392457, 10.713662366666668), (59.9392457, 10.7228439), (59.9392457, 10.732025433333334), (59.9392457, 10.741206966666667), (59.9392457, 10.7503885), (59.9392457, 10.759570033333334), (59.9392457, 10.768751566666667), (59.9392457, 10.7779331), (59.9392457, 10.787114633333333), (59.9392457, 10.796296166666666), (59.9392457, 10.8054777), (59.9392457, 10.814659233333334), (59.93594363333333, 10.686117766666667), (59.93594363333333, 10.6952993), (59.93594363333333, 10.704480833333333), (59.93594363333333, 10.713662366666668), (59.93594363333333, 10.7228439), (59.93594363333333, 10.732025433333334), (59.93594363333333, 10.741206966666667), (59.93594363333333, 10.7503885), (59.93594363333333, 10.759570033333334), (59.93594363333333, 10.768751566666667), (59.93594363333333, 10.7779331), (59.93594363333333, 10.787114633333333), (59.93594363333333, 10.796296166666666), (59.93594363333333, 10.8054777), (59.93594363333333, 10.814659233333334), (59.93264156666667, 10.686117766666667), (59.93264156666667, 10.6952993), (59.93264156666667, 10.704480833333333), (59.93264156666667, 10.713662366666668), (59.93264156666667, 10.7228439), (59.93264156666667, 10.732025433333334), (59.93264156666667, 10.741206966666667), (59.93264156666667, 10.7503885), (59.93264156666667, 10.759570033333334), (59.93264156666667, 10.768751566666667), (59.93264156666667, 10.7779331), (59.93264156666667, 10.787114633333333), (59.93264156666667, 10.796296166666666), (59.93264156666667, 10.8054777), (59.93264156666667, 10.814659233333334), (59.9293395, 10.686117766666667), (59.9293395, 10.6952993), (59.9293395, 10.704480833333333), (59.9293395, 10.713662366666668), (59.9293395, 10.7228439), (59.9293395, 10.732025433333334), (59.9293395, 10.741206966666667), (59.9293395, 10.7503885), (59.9293395, 10.759570033333334), (59.9293395, 10.768751566666667), (59.9293395, 10.7779331), (59.9293395, 10.787114633333333), (59.9293395, 10.796296166666666), (59.9293395, 10.8054777), (59.9293395, 10.814659233333334), (59.926037433333335, 10.686117766666667), (59.926037433333335, 10.6952993), (59.926037433333335, 10.704480833333333), (59.926037433333335, 10.713662366666668), (59.926037433333335, 10.7228439), (59.926037433333335, 10.732025433333334), (59.926037433333335, 10.741206966666667), (59.926037433333335, 10.7503885), (59.926037433333335, 10.759570033333334), (59.926037433333335, 10.768751566666667), (59.926037433333335, 10.7779331), (59.926037433333335, 10.787114633333333), (59.926037433333335, 10.796296166666666), (59.926037433333335, 10.8054777), (59.926037433333335, 10.814659233333334), (59.922735366666664, 10.686117766666667), (59.922735366666664, 10.6952993), (59.922735366666664, 10.704480833333333), (59.922735366666664, 10.713662366666668), (59.922735366666664, 10.7228439), (59.922735366666664, 10.732025433333334), (59.922735366666664, 10.741206966666667), (59.922735366666664, 10.7503885), (59.922735366666664, 10.759570033333334), (59.922735366666664, 10.768751566666667), (59.922735366666664, 10.7779331), (59.922735366666664, 10.787114633333333), (59.922735366666664, 10.796296166666666), (59.922735366666664, 10.8054777), (59.922735366666664, 10.814659233333334), (59.9194333, 10.686117766666667), (59.9194333, 10.6952993), (59.9194333, 10.704480833333333), (59.9194333, 10.713662366666668), (59.9194333, 10.7228439), (59.9194333, 10.732025433333334), (59.9194333, 10.741206966666667), (59.9194333, 10.7503885), (59.9194333, 10.759570033333334), (59.9194333, 10.768751566666667), (59.9194333, 10.7779331), (59.9194333, 10.787114633333333), (59.9194333, 10.796296166666666), (59.9194333, 10.8054777), (59.9194333, 10.814659233333334), (59.91613123333333, 10.686117766666667), (59.91613123333333, 10.6952993), (59.91613123333333, 10.704480833333333), (59.91613123333333, 10.713662366666668), (59.91613123333333, 10.7228439), (59.91613123333333, 10.732025433333334), (59.91613123333333, 10.741206966666667), (59.91613123333333, 10.7503885), (59.91613123333333, 10.759570033333334), (59.91613123333333, 10.768751566666667), (59.91613123333333, 10.7779331), (59.91613123333333, 10.787114633333333), (59.91613123333333, 10.796296166666666), (59.91613123333333, 10.8054777), (59.91613123333333, 10.814659233333334), (59.91282916666667, 10.686117766666667), (59.91282916666667, 10.6952993), (59.91282916666667, 10.704480833333333), (59.91282916666667, 10.713662366666668), (59.91282916666667, 10.7228439), (59.91282916666667, 10.732025433333334), (59.91282916666667, 10.741206966666667), (59.91282916666667, 10.7503885), (59.91282916666667, 10.759570033333334), (59.91282916666667, 10.768751566666667), (59.91282916666667, 10.7779331), (59.91282916666667, 10.787114633333333), (59.91282916666667, 10.796296166666666), (59.91282916666667, 10.8054777), (59.91282916666667, 10.814659233333334), (59.9095271, 10.686117766666667), (59.9095271, 10.6952993), (59.9095271, 10.704480833333333), (59.9095271, 10.713662366666668), (59.9095271, 10.7228439), (59.9095271, 10.732025433333334), (59.9095271, 10.741206966666667), (59.9095271, 10.7503885), (59.9095271, 10.759570033333334), (59.9095271, 10.768751566666667), (59.9095271, 10.7779331), (59.9095271, 10.787114633333333), (59.9095271, 10.796296166666666), (59.9095271, 10.8054777), (59.9095271, 10.814659233333334), (59.906225033333335, 10.686117766666667), (59.906225033333335, 10.6952993), (59.906225033333335, 10.704480833333333), (59.906225033333335, 10.713662366666668), (59.906225033333335, 10.7228439), (59.906225033333335, 10.732025433333334), (59.906225033333335, 10.741206966666667), (59.906225033333335, 10.7503885), (59.906225033333335, 10.759570033333334), (59.906225033333335, 10.768751566666667), (59.906225033333335, 10.7779331), (59.906225033333335, 10.787114633333333), (59.906225033333335, 10.796296166666666), (59.906225033333335, 10.8054777), (59.906225033333335, 10.814659233333334)]
#transportType = "car" #transit, bicycling
#writeToFile = True
#run(coordVector, transportType, writeToFile)