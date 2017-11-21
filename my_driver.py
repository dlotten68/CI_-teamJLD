from pytocl.driver import Driver
from pytocl.car import State, Command
import sklearn
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor

#from neuralnet1 import *
# prediction = model.predict(hoi)
# prediction = pd.DataFrame(prediction)
# prediction.columns = ['ACCELERATION','BRAKE','STEERING']
# prediction*sdy+meany
class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    #...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    NN1 = joblib.load("/home/student/Documents/torcs-server/torcs-client/NN1.pkl")

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()
        NN1 = joblib.load("/home/student/Documents/torcs-server/torcs-client/NN1.pkl")
        speed = (carstate.speed_x**2+carstate.speed_y**2+carstate.speed_z**2)**.5
        prediction = NN1.predict([speed,
        carstate.distance_from_center,
        castate.angle,
        carstate.distances_from_edge].values.reshape(1,-1))
        prediction = pd.DataFrame(prediction)
        prediction.columns = ['ACCELERATION','BRAKE','STEERING']
        # prediction*sdy+meany
        command.steering = self.steering_ctrl.control(
            prediction[['STEERING']],
            carstate.current_lap_time
        )

        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 80

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
