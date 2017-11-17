#from pytocl.driver import Driver
#from pytocl.car import State, Command
from neuralnet1 import *

prediction = model.predict(hoi)
prediction = pd.DataFrame(prediction)
prediction.columns = ['ACCELERATION','BRAKE','STEERING']
print(prediction*sdy+meany)

#class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command
