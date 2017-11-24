from pytocl.driver import Driver
from pytocl.car import State, Command
import sklearn
import pandas as pd
import numpy as np
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

        prediction = self.makePrediction(carstate)

        self.steer(carstate, prediction , command)

        v_x = 80

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command

    def steer(self, carstate, prediction, command):
        print(prediction[0,2])
        steering = prediction[0,2]
        command.steering = self.steering_ctrl.control(
            steering,
            carstate.current_lap_time
        )
