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
        self.steer(carstate, 0.05, command)

        #ACC_LATERAL_MAX = 6400 * 5
        #if(command.steering != 0):
        #    v_x = min(120, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
    #    else:
        v_x = 100

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
        #print(carstate.angle)
        # if(abs(carstate.angle) > 60):
        #     command.gear = -1
        # if(carstate.gear == -1 and abs(carstate.angle)<60 ):
        #     command.gear = 0
        # if(abs(carstate.angle)> 60):
        #     self.probeer(carstate, target_track_pos, command)
        # if(abs(carstate.angle) > 60 and carstate.speed_x/MPS_PER_KMH < 0.5):
        #     command.gear = -1
        #     self.iAmStuck(carstate,target_track_pos, command)
        # # elif(abs(carstate.speed_y) > carstate.speed_x * 0.25):
        #     self.preventDrifting(carstate, target_track_pos, command, 2)
        if(abs(carstate.distance_from_center) > 0.7):
            if(abs(carstate.distance_from_center) <= 1.5):
                self.adjustedSteering(carstate, target_track_pos, command)
            else:
                # command.brake = 1
                self.outsideTrack(carstate,target_track_pos, command)
        else:
            self.standardSteering(carstate, target_track_pos,command)


    # def preventDrifting(self, carstate, target_track_pos, command):
    #     steering = (carstate.angle/180)/mode
    #     print("Prevent drifting")
    #     print(steering)
    #     # print(carstate.angle)
    #     command.steering = self.steering_ctrl.control(
    #         steering,
    #         carstate.current_lap_time
    #     )

    def standardSteering(self, carstate, target_track_pos, command):
        steering = target_track_pos - (self.range_finder_angles[carstate.distances_from_edge.index(max(carstate.distances_from_edge))])
        steering = steering / 90
        print("Standard")
        print(steering)
        # if(abs(steering) > 0.5):
        command.steering = self.steering_ctrl.control(
                steering,
                carstate.current_lap_time
        )

    def adjustedSteering(self, carstate, target_track_pos, command):
        steering = target_track_pos - carstate.distance_from_center
        print("adjust steering ")
        # print(steering)
        # if(steering > 1):
        #     steering = 1
        # elif(steering < -1):
        #     steering = -1
        print(steering)
        # if(abs(steering) > 0.5):
        command.steering = self.steering_ctrl.control(
                steering,
                carstate.current_lap_time
        )

    def iAmStuck(self, carstate, target_track_pos, command):
        # 128 moet hij -angle/180 doen. dus naar rechts, want staat links vast met scherpe hoek tegen boarding?
        # -137 staat hij ook links vast..
        print("I am stuck")
        self.adjustedSteering(carstate, target_track_pos, command)
        command.gear = 0
        # command.gear = -1
        # self.adjustedSteering(carstate, target_track_pos, command)

    def outsideTrack(self, carstate, target_track_pos, command):
        print("outside track")
        print(carstate.angle)
        if(carstate.angle < 0):
            command.gear = -1
            print(carstate.gear)
        else:
            self.adjustedSteering(carstate, 0.8, command)

    # def probeer(self, carstate, target_track_pos, command):
    #     print("probeer")
    #     command.gear = 0
    #     self.adjustedSteering(carstate, target_track_pos, command)
