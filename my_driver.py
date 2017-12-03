from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
import sklearn
import pandas as pd
import numpy as np
import math
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor

stuckCounter = 0
steerPrevious  = 0
maxSteer = 0
minSteer = 0
recover = 0
wrongwaycounter = 0
berm = 0
bermsolve = 0


class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    #...

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        global recover
        command = Command()

        print("===================================================")
        print("recover " + repr(recover))
        # Met heuristiek:
        corner = self.cornerLearner(carstate)
        print("corner: ", end=' ')
        print(corner, end =' ')
        print("degrees", end =' ')
        self.steer(carstate, 0.0, corner, command)
        v = self.speedCorner(corner, carstate)
        if(abs(carstate.distance_from_center) > 1):
            v = 40
        self.accelerate(carstate, v, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command

    # Deze is met heuristiek
    def steer(self, carstate, target_track_pos, corner, command):
        global berm
        global stuckCounter
        global steerPrevious
        global wrongwaycounter
        global recover
        print("carstate angle: "+ repr(carstate.angle))
        print("carstate distance from center: " +repr(carstate.distance_from_center))

        maxDistance = max(carstate.distances_from_edge)
        maxDistanceIndex = carstate.distances_from_edge.index(max(carstate.distances_from_edge))

        if(abs(carstate.distance_from_center)>1 and
        abs(carstate.distance_from_center)<1.1 and
        abs(carstate.wheel_velocities[0]-carstate.wheel_velocities[1])>100 and
        abs(carstate.wheel_velocities[2]-carstate.wheel_velocities[3])>100):
            berm += 1
            print("Berm" + repr(berm))
        else:
             berm = 0

        if(abs(carstate.angle)>120):
            wrongwaycounter +=1
        else:
            wrongwaycounter = 0
        print("wrongwaycounter "+ repr(wrongwaycounter))

        if(carstate.gear == -1 and carstate.distance_from_center*carstate.angle > 0):
            command.gear = 1
        print("stuckCounter " + repr(stuckCounter))
        speed = (carstate.speed_x**2+carstate.speed_y**2+carstate.speed_z**2)**.5
        if(abs(carstate.angle) > 20 and
        carstate.speed_x<10 and
        abs(carstate.distance_from_center) > 0.5) and
        (carstate.distance_from_center*carstate.angle)<0.0):
            stuckCounter = stuckCounter + 1
        else:
            stuckCounter = 0

        if(stuckCounter >= 200) or wrongwaycounter>100:
            recover = 10000
            self.iAmStuck(carstate,target_track_pos, command)
        elif(berm >100):
            recover = 1000
            self.solveberm(carstate, target_track_pos, command)
        elif(abs(carstate.distance_from_center) > 0.8):
            if(abs(carstate.distance_from_center) > 1):
                recover = 10000
                self.offTrack(carstate,target_track_pos, command)
            else:
                self.adjustedSteering(carstate, target_track_pos, command)
        else:
            self.standardSteering(carstate, target_track_pos, corner, command)

    def solveberm(self, carstate, target_track_pos, command):
        global recover
        global bermsolve
        if carstate.distance_from_center <0.0:
            steering = 1
        else:
            steering = -1
        bermsolve = 1

    def adjustedSteering(self, carstate, target_track_pos, command):
        global steerPrevious
        global recover
        steering = (target_track_pos - carstate.distance_from_center)/5
        if recover >0:
            recover -=1

        print("adjust steering ")
        print("steering:" + repr(steering))
        steerPrevious = steering
        command.steering = self.steering_ctrl.control(
                steering,
                carstate.current_lap_time
        )

    def iAmStuck(self, carstate, target_track_pos, command):
        global steerPrevious
        global recover
        # 128 moet hij -angle/180 doen. dus naar rechts, want staat links vast met scherpe hoek tegen boarding?
        # -137 staat hij ook links vast..
        print("I am stuck")
        print("carstate.gear:" + repr(carstate.gear))
        if recover < 0:
            recover -=1

        steering = -carstate.angle / 45
        print("steering:" + repr(steering))
        steerPrevious = steering
        command.steering = self.steering_ctrl.control(
            steering,
            carstate.current_lap_time
        )

    def offTrack(self, carstate, target_track_pos, command):
        global steerPrevious
        global recover
        print("outside track")
        steering = (carstate.angle - 30 * carstate.distance_from_center)/45
        if recover>0:
            recover -=1
        print("steering:" + repr(steering))
        steerPrevious = steering
        command.steering = self.steering_ctrl.control(
            steering,
            carstate.current_lap_time
        )
        print("wheelspin: " +repr(carstate.wheel_velocities))

    def accelerate(self, carstate, target_speed, command):
        global stuckCounter
        global recover
        global bermsolve
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
            if abs(carstate.distance_from_center) >= 1.0:
                # off track, reduced grip:
                acceleration = min(0.5, acceleration)

            if(abs(carstate.distance_from_center) < 0.8 and abs(carstate.angle) > 30):
                acceleration = min(0.3, acceleration)

            if carstate.gear ==-1:
                acceleration = min(0.2, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

            if recover > 0:
                acceleration = min(0.4, acceleration)
                if bermsolve == 1:
                    acceleration = 0
                    bermsolve = 0

            command.accelerator = min(acceleration, 1)
        print("acceleration:" + repr(acceleration))

        if carstate.rpm < 2500 :
            if(carstate.gear != -1):
                command.gear = carstate.gear - 1
            if(command.gear == -2):
                print("+++++++++++++++++++++++++++++++++++++++++++")
                system.exit()

        if(stuckCounter >= 200): #or wrongwaycounter>100:
            command.gear = -1

        if not command.gear:
            command.gear = carstate.gear or 1


    def standardSteering(self, carstate, target_track_pos, corner, command):
        # Target track position is the position of the track on which you orientate
        # that is zero for now
        global steerPrevious
        global recover
        print("Standard")
        print("Maxdist "+repr(max(carstate.distances_from_edge)))
        # range_finder_angles are all angles -90, ..., 0, ..., 90
        #                -90-75-60-45-30, -20,  -15,  -10, -5,
        steeringDegrees = [1, 1, 1, 1, 1, 0.75, 0.63, 0.5, 0.25, \
				    0,-0.25,-0.5,-0.63,-0.75,-1,-1,-1,-1,-1]
	    #			0,    5,  10,   15,  20, 30,45,60,75,90
        # Find the angle with farest distance closest to zero:
        maxIndices = [i for i, j in enumerate(carstate.distances_from_edge) \
            if j == max(carstate.distances_from_edge)]
        angles = [(i, self.range_finder_angles[i]) for i in maxIndices]
        angle = min(angles, key=lambda x:abs(x[1])) # Tuple (index, angle)
        if(angle[1] >= 40):
            print("Maximum steering right")
            steering = -1;
        elif(angle[1] <= -40):
            print("Maximum steering left")
            steering = 1;
        else:
            # Steer is computed using distance measured by the track sensor with
            # maximum distance (and its two adjacent sensors)
            steering = steeringDegrees[angle[0]]
        if(abs(corner) > 35 and abs(corner) < 45):
            steering = steering*2 # Slow
        elif(abs(corner) >= 45):
            steering = steering*3 # Hairpin
        print("steering:" + repr(steering))
        steerPrevious = steering
        if recover > 0:
            recover -=50
        if recover < 0:
            recover = 0
        command.steering = self.steering_ctrl.control(
                steering,
                carstate.current_lap_time
        )

    def cornerLearner(self, carstate):
        N = len(carstate.distances_from_edge)
        border = np.zeros([N,2])
        for i in range(N):
            border[i,:] = [-math.cos(i*math.pi/18)*carstate.distances_from_edge[i],
                           math.sin(i*math.pi/18)*carstate.distances_from_edge[i]]
        maxIndex = carstate.distances_from_edge.index(max(carstate.distances_from_edge))
        minIndex = carstate.distances_from_edge.index(max(carstate.distances_from_edge))

        l = np.zeros([max(minIndex-1,0),2])
        for i in range(minIndex-1):
            dist = np.linalg.norm(border[i+1,:]-border[i,:])
            l[i,:] = (border[i+1,:]-border[i,:])/dist

        r = np.zeros([max(N-maxIndex-2,0),2])
        for i in range(N-maxIndex-2):
            dist = np.linalg.norm(border[N-i-2]-border[N-1-i])
            r[i,:] = (border[N-i-2]-border[N-1-i])/dist

        corner = 0
        for i in range(minIndex-1-1):
            corner = corner + math.acos(np.dot(l[i,:],l[i+1,:]))*180/math.pi*\
            np.sign(l[i,0]*l[i+1,1]-l[i,1]*l[i+1,0])

        for i in range(N-1-maxIndex-1-1):
            corner = corner + math.acos(np.dot(r[i,:],r[i+1,:]))*180/math.pi*\
            np.sign(r[i,0]*r[i+1,1]-r[i,1]*r[i+1,0])

        return corner

    def speedCorner(self, corner, carstate):
        trackWidth = carstate.distances_from_edge[0]+carstate.distances_from_edge[18]
        if(abs(corner) < 3 and max(carstate.distances_from_edge) < 90):
            v = 200 # Straight Approacing Corner
            print("Straight AC")
        elif(abs(corner) < 3 ):
            v = 350 # Straight
            print("Straight")
        elif(abs(corner) < 20):
            v = 200 # Full speed corner
            print("Full")
        elif(abs(corner) < 35 and max(carstate.distances_from_edge) > trackWidth*5):
            v = 155 # Medium far distance
            print("MediumFar")
        elif(abs(corner) < 35):
            v = 180 # Medium
            print("Medium")
        elif(abs(corner) < 45):
            v = 150 # Slow
            print("Slow")
        else:
            v = 79 # Hairpin
            print("Hairpin")
        return v
