from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
#from pytocl.Opponents import State
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
closest = 0
unstuckCounter = 0

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

    def avoidance(self, steerkind, steering, carstate):
        min_range = 201
        min_sensor = 40
        hard = 0.3
        firm = 0.25
        steerplan = steering # store our original steerplan

        for sensor in range(11,23): # kijk +60 tot -60 graden vooruit
            if(carstate.opponents[sensor]<min_range):
                min_sensor = sensor
            min_range = min(min_range, carstate.opponents[sensor])
        if(min_sensor == 12 and min_range/carstate.speed_x >= 0.30): # closest opp 60+ degrees left and close enough
            steering -= 0.1 # steer a little to the right
        if(min_sensor == 13 and min_range/carstate.speed_x >= 0.50 or min_sensor == 14 and min_range/carstate.speed_x>=0.50): #closest opp 50 or 40 degrees left and close enough
            steering -= 0.12
        if(min_sensor == 15 and min_range/carstate.speed_x >= 0.75): #closest opp 30 degrees left and close enough
            steering -= 0.13
        if(min_sensor == 16 and min_range/carstate.speed_x >= 0.75): #closest opp 20 degrees left and close enough
            steering -= 0.14
        if(min_sensor == 17 and min_range/carstate.speed_x >= 1): #closest opp 10 degrees left and close enough
            steering -= 0.15

        if(min_sensor == 18 and min_range/carstate.speed_x >= 1): #closest opp 10 degrees right and close enough
            steering += 0.15  # steer to the left
        if(min_sensor == 19 and min_range/carstate.speed_x >= 0.75): #closest opp 20 degrees right and close enough
            steering += 0.14
        if(min_sensor == 20 and min_range/carstate.speed_x >= 0.75): #closest opp 30 degrees right and close enough
            steering += 0.13
        if(min_sensor == 21 and min_range/carstate.speed_x >= 0.50 or min_sensor == 22 and min_range/carstate.speed_x>=0.50): #closest opp 50 or 40 degrees right and close enough
            steering += 0.12
        if(min_sensor == 23 and min_range/carstate.speed_x >= 0.30): # closest opp 60+ degrees right and close enough
            steering -= 0.1 # steer a little to the right

        # above were the overtake scenarios ... these wlll be overwritten if any of the following are true
        if(min_sensor == 14 and min_range <= 10): # closest opp 30+ degrees left and close
            steering -= firm # steer to the right
        if(min_sensor == 15 and min_range <= 10): # closest opp 20-30 degrees left and close
            steering -= firm # steer to the right
        if(min_sensor == 16 and min_range <= 10): # closest opp 10-20 degrees left and close
            steering -= firm # steer to the right

        meeOfTegen = 0
        if(carstate.distance_from_center*steering>0):
            meeOfTegen = -1
        else:
            meeOfTegen = 1

        if(min_sensor == 17 and min_range<=10 or min_sensor == 18 and min_range <= 10): # closest opp dead ahead and close
        #    steering -= 0.1 # steer a little to the right
            print("collision imminent")
            if(steerkind == 1):
                if(steering < 0):
                    steering -= 0.3*meeOfTegen # if we are on edge of track do not steer offTrack
                else:
                    steering += 0.3*meeOfTegen
            elif(steerkind == 0): #( steerkind == 0 is normalsteering)
                if(steering < 0):
                    steering -= 0.3
                else:
                    steering += 0.3
        if(min_sensor == 19 and min_range <= 10): # closest opp 30+ degrees left and close
            steering += 0.25 # steer to the right
        if(min_sensor == 20 and min_range <= 10): # closest opp 20-30 degrees left and close
            steering += 0.25 # steer to the right
        if(min_sensor == 21 and min_range <= 10): # closest opp 10-20 degrees left and close
            steering += 0.25 # steer to the right
# we will not go off track to avoid a collision. If the new steer will get us off track, we will revert to our original steer
        if(steerkind == 1 and carstate.distance_from_center*steering>0): #if we are on the edge and intend to steer more in that direction, revert to plan
            steering = steerplan
#         closest = carstate.opponents.index(min(carstate.opponents)) # dit is een sensor nummer
#         meeOfTegen = 0
#         print("distance_to =" + repr(min(carstate.opponents)))
#         d = -1*math.cos((math.pi)/180 * closest*10+5)
#         print("distanceto =" + repr(min(carstate.opponents)))
#         distanceto = min(carstate.opponents)
#         print("opp sensor =" + repr(carstate.opponents.index(min(carstate.opponents))))
#         closest = carstate.opponents.index(min(carstate.opponents)) # sensor nummer
#         dX = -distanceto*math.cos((math.pi)/180 * closest*10+5) # dit is de x afstand (langs de rijrichting)
# #       front collision avoidance
#         mindist = 200
#         for sensor in range(16,19):
#              mindist = min(mindist, carstate.opponents[sensor])
#         sensor18reading_X = -mindist*math.cos((math.pi)/180 * 18*10)
#         if(carstate.distance_from_center*steering>0):
#             meeOfTegen = -1
#         else:
#             meeOfTegen = 1
#         if(sensor18reading_X <= 15):
#             print("collision imminent" + repr(sensor18reading_X))
#             if(steerkind == 1):
#                 if(steering < 0):
#                     steering -= 0.3*meeOfTegen
#                 else:
#                     steering+=0.3*meeOfTegen
#             elif(steerkind == 0): #(dus steerkind == 0 is normalsteering)
#                 if(steering < 0):
#                     steering -= 0.3
#                 else:
#                     steering += 0.3

        print("minsensor = " + repr(min_sensor))
        print("mindist = " + repr(min_range))
        return steering



#        print("x dist =" + repr(dX))
#        d = "none"
#        if(dddd < -7.5):
#            d = "back"
#        elif(dddd< 7.5):
#            d = "parallel"
#        elif(dddd<35):
#            d = "close"
#        print("forward distance = " + d)






    # Deze is met heuristiek
    def steer(self, carstate, target_track_pos, corner, command):
        global closest
        global berm
        global stuckCounter
        global steerPrevious
        global wrongwaycounter
        global recover
        global unstuckCounter
        #print("carstate angle: "+ repr(carstate.angle))
        #print("carstate distance from center: " +repr(carstate.distance_from_center))

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

        if(abs(carstate.angle)>120 and carstate.gear >0):
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
        abs(carstate.distance_from_center) > 0.5) and (carstate.distance_from_center*carstate.angle)<0.0:
            stuckCounter = stuckCounter + 1
        else:

            stuckCounter = 0



        if(stuckCounter >= 200 or wrongwaycounter>100):
            recover = 10000
            self.iAmStuck(carstate,target_track_pos, command)
        # elif(unstuckCounter>0):
        #     unstuckCounter = max(0,unstuckCounter-1)
        #     print("unstuck " + repr(unstuckCounter))
        #     self.iAmStuck(carstate,target_track_pos, command)

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
        self.accelerate(carstate, 100, command)
        print("gelukt")

    def adjustedSteering(self, carstate, target_track_pos, command):
        global steerPrevious
        global recover
        steering = (target_track_pos - carstate.distance_from_center)/5
        if recover >0:
            recover -=1

        print("adjust steering ")
#        print("steering:" + repr(steering))
        steerPrevious = steering
        ############################################
        if(carstate.speed_x > 0): steering = self.avoidance(1, steering, carstate)

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
        if recover > 0:
            recover -=1

        steering = -carstate.angle / 45
        print("steering:" + repr(steering))
        steerPrevious = steering
        command.steering = self.steering_ctrl.control(
            steering,
            carstate.current_lap_time
        )
        self.accelerate(carstate, 30, command)
        print("gelukt2")



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
        self.accelerate(carstate, 50, command)
        print("gelukt3")


    def accelerate(self, carstate, target_speed, command):
        global stuckCounter
        global recover
        global bermsolve
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - math.sqrt(math.pow(carstate.speed_x,2)+math.pow(carstate.speed_y,2)+math.pow(carstate.speed_z,2))
        print("speederror: ", repr(speed_error))
        AB = 2*(.5-1/(1+math.exp(speed_error)))
        print("AB " + repr(AB))
        acceleration = self.acceleration_ctrl.control(
            AB,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            #if abs(carstate.distance_from_center) >= 1.0 or abs(carstate.distance_from_center) < 0.8 and abs(carstate.angle) > 30 or carstate.gear ==-1 or recover > 0:
            if(recover>0):
                acceleration = min(0.4, acceleration)
            #if abs(carstate.distance_from_center) >= 1.0:
                # off track, reduced grip:
            #    acceleration = min(0.5, acceleration)

            #if(abs(carstate.distance_from_center) < 0.8 and abs(carstate.angle) > 30):
                #acceleration = min(0.4, acceleration)

        #    if carstate.gear ==-1:
        #        acceleration = min(0.25, acceleration)

        #    command.accelerator = min(acceleration, 1)

        #    if recover > 0:
        #        acceleration = min(0.3, acceleration)
            if bermsolve == 1:
                #acceleration = 0
                bermsolve = 0

            command.accelerator = min(acceleration, 1)
        if(acceleration<0):
            command.accelerator = 0
            command.brake = min(abs(acceleration),1)*0.48


        print("acceleration:" + repr(acceleration))

        if ((carstate.gear == 1 or carstate.gear == 2 or carstate.gear == 3) and carstate.rpm >= 9000):
            command.gear = carstate.gear + 1
        elif ((carstate.gear == 4 or carstate.gear == 5) and carstate.rpm >= 8000):
            command.gear = carstate.gear + 1
            # Shift down
        elif ((carstate.gear == 2 or carstate.gear == 3 or carstate.gear == 4) and carstate.rpm <= 3000):
            command.gear = carstate.gear - 1
        elif ((carstate.gear == 5 or carstate.gear == 6) and carstate.rpm <= 3500):
            command.gear = carstate.gear - 1

        if(stuckCounter >= 200 or wrongwaycounter>100) and carstate.gear >= -1: #or wrongwaycounter>100:
            command.gear = -1


        if not command.gear:
            command.gear = carstate.gear or 1


    def standardSteering(self, carstate, target_track_pos, corner, command):
        # Target track position is the position of the track on which you orientate
        # that is zero for now
        global steerPrevious
        global recover
        print("Standard")
#        print("Maxdist "+repr(max(carstate.distances_from_edge)))
        # range_finder_angles are all angles -90, ..., 0, ..., 90
        #                -90-75-60-45-30, -20,  -15,  -10, -5,
        steeringDegrees = [1, .9, .80, .70, .60, 0.50, 0.45, 0.30, 0.10, \
				    0,-0.10,-0.30,-0.45,-0.50,-.60,-.70,-.8,-.9,-1]
	    #			0,    5,  10,   15,  20, 30,45,60,75,90
        # Find the angle with farest distance closest to zero:
        maxIndices = [i for i, j in enumerate(carstate.distances_from_edge) \
            if j == max(carstate.distances_from_edge)]
        angles = [(i, self.range_finder_angles[i]) for i in maxIndices]
        angle = min(angles, key=lambda x:abs(x[1])) # Tuple (index, angle)
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
            if(abs(carstate.angle >20)):
                 steering = carstate.angle/30
            recover -=50
        if recover < 0:
            recover = 0
    ###############################################3
        if(carstate.speed_x>0): steering = self.avoidance(0, steering, carstate)
#        print("cs.curr lapt:" + repr(carstate.current_lap_time))

#        print("steering check:" + repr(steering))
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
        indexList = [i for i, j in enumerate(carstate.distances_from_edge) if j == max(carstate.distances_from_edge)]
        maxIndex = indexList[-1]
        minIndex = indexList[0]

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
        global recover
        trackWidth = carstate.distances_from_edge[0]+carstate.distances_from_edge[18]
        if( recover > 0):
            v = 40
        elif(abs(corner) < 3 and max(carstate.distances_from_edge) < 90):
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
