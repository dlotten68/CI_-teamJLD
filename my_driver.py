from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
import sklearn
import pandas as pd
import numpy as np
import math
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor

NN1 = joblib.load("/home/student/Documents/torcs-server/torcs-client/NN1.pkl")
#from neuralnet1 import *
# prediction = model.predict(hoi)
# prediction = pd.DataFrame(prediction)
# prediction.columns = ['ACCELERATION','BRAKE','STEERING']
# prediction*sdy+meany
stuckCounter = 0
steerHOI  = 0
maxSteer = 0
minSteer = 0


class MyDriver(Driver):
    #NN1 = joblib.load("/home/student/Documents/torcs-server/torcs-client/NN1.pkl")
    # Override the `drive` method to create your own driver
    #...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return commandt


    def makePrediction(self, carstate):
        speed = (carstate.speed_x**2+carstate.speed_y**2+carstate.speed_z**2)**.5
        lEdges = list(carstate.distances_from_edge)
        predictionInput = [speed, carstate.distance_from_center, carstate.angle]
        predictionInput.extend(lEdges)

        predictionInput = pd.Series(predictionInput).values.reshape(1,-1)
        meanPredictionInput = np.mean(predictionInput)
        sdPredictionInput = np.std(predictionInput)
        scaledPredictionInput = (predictionInput - meanPredictionInput)/sdPredictionInput

        prediction = NN1.predict(scaledPredictionInput)
        meanPrediction = np.mean(prediction)
        sdPrediction = np.std(prediction)
        prediction = prediction * sdPrediction + meanPrediction
        return prediction

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()

        #     # Met NN1.pkl:
        # prediction = self.makePrediction(carstate)
	    # self.steer(carstate, prediction , command)
            # Met heuristiek:
        self.steer(carstate, 0.0, command)

        #ACC_LATERAL_MAX = 6400 * 5
        #if(command.steering != 0):
        #    v_x = min(120, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
    #    else:
        v_x = 80
        if(abs(carstate.distance_from_center) > 1):
            v_x = 30

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command

    #        # Deze is voor met NN1.pkl
    # def steer(self, carstate, prediction, command):
    #     print(prediction[0,2])
    #     steering = prediction[0,2]
    #     command.steering = self.steering_ctrl.control(
    #     steering,
    #     carstate.current_lap_time
    #     )

    #        # Deze is met heuristiek
    def steer(self, carstate, target_track_pos, command):
        print("===================================================")
        global stuckCounter
        global steerHOI
        print(carstate.angle)
        print(carstate.distance_from_center)

        #distancesArray = carstate.distances_from_edge
        maxDistance = max(carstate.distances_from_edge)
        maxDistanceIndex = carstate.distances_from_edge.index(max(carstate.distances_from_edge))
        # leftIndices = list(range(maxDistanceIndex - 1))
        # rightIndices = list(range(maxDistanceIndex+1,19))
        # r0 = (self.range_finder_angles[leftIndices[0]] - self.range_finder_angles[leftIndices[1]])/abs((self.range_finder_angles[leftIndices[0]] - self.range_finder_angles[leftIndices[1]]))
        #print(distancesArray)
        # print(maxDistance)
        # print(maxDistanceIndex)
        # print(leftIndices)
        # print(rightIndices)
        # print(r0)



        # if(abs(carstate.angle) > 60):
        #     command.gear = -1
        # if(carstate.gear == -1 and abs(carstate.angle)<60 ):
        #     command.gear = 0
        # if(abs(carstate.angle)> 60):
        #     self.probeer(carstate, target_track_pos, command
        if(carstate.gear == -1 and carstate.distance_from_center*carstate.angle > 0):
            command.gear = 1
        print(stuckCounter)
        speed = (carstate.speed_x**2+carstate.speed_y**2+carstate.speed_z**2)**.5
        # if(speed > 20 and speed < 50):
        #     maxSteer = 0.5
        #     minSteer = -0.5
        # elif(speed > 50):
        #     maxSteer = 0.1
        #     minSteer = -0.1
        if(carstate.angle > 30 and carstate.distance_from_center < -0.5) or (carstate.angle < -30 and carstate.distance_from_center > 0.5):
            stuckCounter = stuckCounter + 1
        # if(abs(carstate.angle) > 30 and abs(carstate.distance_from_center) >= 0.5 or speed < 10):
        #     stuckCounter = stuckCounter + 1
        else:
            stuckCounter = 0
        #     command.gear = -1
        if(stuckCounter >= 500):
            self.iAmStuck(carstate,target_track_pos, command)
        # # elif(abs(carstate.speed_y) > carstate.speed_x * 0.25):
        #     self.preventDrifting(carstate, target_track_pos, command, 2)
        elif(abs(carstate.distance_from_center) > 0.8):
            if(abs(carstate.distance_from_center) > 1):
                self.offTrack(carstate,target_track_pos, command)
            else:
                self.adjustedSteering(carstate, target_track_pos, command)
        elif(abs(carstate.angle) >10 ):
            self.allignWithCenter(carstate, target_track_pos, command)
        else:
            self.standardSteering(carstate, target_track_pos,command)


    def allignWithCenter(self, carstate, target_track_pos, command):
        global steerHOI
        steering = (carstate.angle - 30 * carstate.distance_from_center)/ 45
        print("Allign with center")
        print(steering)
        # print(carstate.angle)
        steerHOI = steering
        command.steering = self.steering_ctrl.control(
            steering,
            carstate.current_lap_time
        )


    def standardSteering(self, carstate, target_track_pos, command):
        global steerHOI
        steering = target_track_pos - (self.range_finder_angles[carstate.distances_from_edge.index(max(carstate.distances_from_edge))])
        steering = steering / 90
        print("Standard")
        print(steering)
        steerHOI = steering
        # if(steering > maxSteer):
        #     steering = maxSteer
        # if(steering < minSteer):
        #     steering = minSteer
        command.steering = self.steering_ctrl.control(
                steering,
                carstate.current_lap_time
        )

    def adjustedSteering(self, carstate, target_track_pos, command):
        steering = target_track_pos - carstate.distance_from_center
        # if(steering > maxSteer):
        #     steering = maxSteer
        # if(steering < minSteer):
        #     steering = minSteer
        print("adjust steering ")
        # if(steering > 1):
        #     steering = 1
        # elif(steering < -1):
        #     steering = -1
        print(steering)
        steerHOI = steering
        command.steering = self.steering_ctrl.control(
                steering,
                carstate.current_lap_time
        )

    def iAmStuck(self, carstate, target_track_pos, command):
        # 128 moet hij -angle/180 doen. dus naar rechts, want staat links vast met scherpe hoek tegen boarding?
        # -137 staat hij ook links vast..
        print("I am stuck")
        print(carstate.gear)
        steering = -carstate.angle / 45

        # if(steering > maxSteer):
        #     steering = maxSteer
        # if(steering < minSteer):
        #     steering = minSteer
        print(steering)
        steerHOI = steering
        command.steering = self.steering_ctrl.control(
            steering,
            carstate.current_lap_time
        )
        # command.gear = -1
        # self.adjustedSteering(carstate, target_track_pos, command)

    def offTrack(self, carstate, target_track_pos, command):
        print("outside track")
        print(carstate.angle)
        steering = (carstate.angle - 30 * carstate.distance_from_center)/45
        # if(steering > maxSteer):
        #     steering = maxSteer
        # if(steering < minSteer):
        #     steering = minSteer
        print(steering)
        steerHoi = steering
        command.steering = self.steering_ctrl.control(
            steering,
            carstate.current_lap_time
        )

    # def probeer(self, carstate, target_track_pos, command):
    #     print("probeer")
    #     command.gear = 0
    #     self.adjustedSteering(carstate, target_track_pos, command)

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 0.8 or carstate.gear == -1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)
            if(abs(carstate.distance_from_center) < 0.8 and abs(carstate.angle) > 20):
                acceleration = min(0.2, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1


        # else:
        #     command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500 :
            if(carstate.gear != -1):
                command.gear = carstate.gear - 1
            if(command.gear == -2):
                print("+++++++++++++++++++++++++++++++++++++++++++")
                system.exit()

        if(stuckCounter >= 500):
            command.gear = -1

        if not command.gear:
            command.gear = carstate.gear or 1
