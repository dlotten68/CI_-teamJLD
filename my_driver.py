from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
import sklearn
import pandas as pd
import numpy as np
import math
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
import pickle
import neat
import gzip

# Global constants
rho_slow = 25 # max car angle to track, speed < rho_speed
rho_fast = 15 # max car angle to track, speed > rho_speed
rho_speed = 120
rho = 0
deviate_steer = 0.4 # max deviation previous steer
max_dist_centre = 0.9 # max distance to center
stuck_dist_centre = 0.5 # Not stuck if you are on track [-0.5, 0.5]
berm_dist_centre = 1.2 # if further then 1.2, you are not in berm
berm_wheel_speed = 100 # Wheel speed difference needed for berm recovery
off_track_constant = 30  # Komt van 1/6 pi in graden
wrong_way_angle = 120 # wrong way if angle is > wrong_way_angle
###########
#(verkort)#
###########
berm_time = 20 # time needed to be in berm before recovery
stuck_time = 150 # time needed to be stuck before recovery

wrongway_time = 100 # time needed to drive wrong way before recovery

###########
#(verkort)#
###########
offtrack_time = 20 # time needed to be off-track before recovery
recover_speed = 17/MPS_PER_KMH # maximum speed to end recovery
speed_correction = 100 # constant to bound steering
max_steer_degrees = 45 # maximum steering angle of program
sensor_max_dist = 200 # max distance sensor of program
center_coef = 10 # Steering to the middle, the higher the more in the middle - van 10 --> 11
recover_time = 1
align_coef = 1
center_coef = 5
furthest_coef = 1
#global variables
recover_counter = 0
stuckCounter = 0
wrongwayCounter = 0
bermCounter = 0
offtrackCounter =0
lastSteering = 0
lastAcceleration = 0
recover = 0
recover_speed_kmh = 20
accelerationPrevious = 0

class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        #global CONSTANTS
        global rho
        global rho_slow
        global rho_fast
        global rho_speed
        global max_dist_centre
        global stuck_dist_centre
        global berm_dist_centre
        global berm_wheel_speed
        global wrong_way_angle
        global berm_time
        global stuck_time
        global wrongway_time
        global offtrack_time
        #global VARIABLES
        global stuckCounter
        global wrongwayCounter
        global bermCounter
        global offtrackCounter
        global recover

        command = Command()
        print("==========================================")



        corner = self.cornerLearner(carstate)
        speed = math.sqrt(math.pow(carstate.speed_x,2)+math.pow(carstate.speed_y,2)+math.pow(carstate.speed_z,2))
        print("speed" + repr(speed))
        if(speed < rho_speed):
            rho = rho_slow
        else:
            rho = rho_fast

        if recover > 0:
            if abs(carstate.distance_from_center) < 0.4 and abs(carstate.angle) < rho:
                recover = 0
            else:
                recover = 1

        if (stuckCounter>stuck_time) or (abs(carstate.angle) > rho and abs(carstate.distance_from_center) >= max_dist_centre and (carstate.distance_from_center*carstate.angle)<0.0) and carstate.speed_x<10/MPS_PER_KMH:
            wrongwayCounter = 0
            bermCounter = 0
            offtrackCounter = 0
            stuckCounter += 1
            print("stuckCounter: " + repr(stuckCounter))
            self.iAmStuck(carstate, speed, command, corner)
        elif (wrongwayCounter > wrongway_time) or (abs(carstate.angle)>wrong_way_angle and carstate.gear >=0):
            stuckCounter = 0
            bermCounter = 0
            offtrackCounter = 0
            wrongwayCounter +=1
            print("wrongwayCounter: " + repr(wrongwayCounter))
            self.wrongway(carstate, speed, command, corner)
        elif (bermCounter>berm_time) or (abs(carstate.distance_from_center)>max_dist_centre and abs(carstate.distance_from_center)<berm_dist_centre and abs(carstate.wheel_velocities[0]-carstate.wheel_velocities[1])>berm_wheel_speed and abs(carstate.wheel_velocities[2]-carstate.wheel_velocities[3])>berm_wheel_speed):
            stuckCounter = 0
            wrongwayCounter = 0
            #offtrackCounter = 0
            bermCounter +=1
            print("bermCounter: " + repr(bermCounter))
            self.bermSolver(carstate, speed, command, corner)
        elif (offtrackCounter>offtrack_time) or (abs(carstate.distance_from_center)>max_dist_centre):
            stuckCounter = 0
            wrongwayCounter = 0
            bermCounter = 0
            offtrackCounter +=1
            print("offtrackCounter: " + repr(offtrackCounter))
            self.offTrack(carstate, speed, command, corner)
        elif(abs(carstate.angle) > rho):
            stuckCounter = 0
            wrongwayCounter = 0
            bermCounter = 0
            offtrackCounter = 0
            print("adjustedSteer")
            self.adjustSteering(carstate, speed, command)
        else:
            stuckCounter = 0
            wrongwayCounter = 0
            bermCounter = 0
            offtrackCounter = 0
            print("standardSteer")
            self.standardSteering(carstate, speed, command, corner)

        if self.data_logger:
            self.data_logger.log(carstate, command)
#            print("dit is de gear command : " + repr(command.gear))
#            print("dit is de accelerator command : " + repr(command.accelerator))

        return command

    def iAmStuck(self, carstate, speed, command, corner):
        global stuckCounter
        global stuck_time
        global rho
        global recover_speed
        global max_dist_centre
        global recover

        recover = recover_time

        if stuckCounter < stuck_time or carstate.distance_from_center*carstate.angle > 0.0: # met je neus de goede kant op isniet stuck
            self.standardSteering(carstate, speed, command, corner)
        else:
            if abs(carstate.angle) > rho and abs(carstate.distance_from_center) > 0.05 and (carstate.distance_from_center*carstate.angle)<0.0:
                print("waar stuck0")
                steering = -carstate.angle / 45
                gear = -1
                brake = 0
                accelerate = 18
                recovery = True
            elif speed > recover_speed and (abs(carstate.angle) < rho):
                print("waar stuck1")
                steering = 0
                gear = 1
                brake = 1
                accelerate = 0
            else:
                print("waar stuck2")
                recovery = False
                stuckCounter = 0
                gear = 1
                brake = 0
                steering = 0
                accelerate = recover_speed_kmh
            self.commands(carstate, brake, accelerate, steering, gear, command)

    def wrongway(self, carstate, speed, command, corner):
        global wrongwayCounter
        global wrongway_time
        global recover_speed
        global wrong_way_angle
        global recover

        if wrongwayCounter < wrongway_time:
            self.standardSteering(carstate, speed, command, corner)
        else:
            recover = recover_time
            if abs(carstate.angle)>wrong_way_angle:
                gear = 1
                steering = carstate.angle / 45
                brake = 0
                accelerate = recover_speed_kmh
            elif speed > recover_speed and abs(carstate.angle)<wrong_way_angle:
#                print("wway 1")
                gear = 1
                steering = 0
                brake = 1
                accelerate = recover_speed_kmh
            else:
#                print("wway 2")

                gear = 1
                wrongwayCounter = 0
                steering = 0
                brake = 0
                accelerate = recover_speed_kmh
            self.commands(carstate, brake, accelerate, steering, gear, command)




    def bermSolver(self, carstate, speed, command, corner):
        global bermCounter
        global berm_time
        global recover_speed
        global berm_dist_centre
        global berm_wheel_speed
        global max_dist_centre
        global recover

        recover = recover_time

        if bermCounter < berm_time:
            #self.standardSteering(carstate, speed, command, corner) # meteen berm in
            niets = 0
        else:
            if(abs(carstate.distance_from_center)>max_dist_centre and abs(carstate.distance_from_center)<berm_dist_centre and abs(carstate.wheel_velocities[0]-carstate.wheel_velocities[1])>berm_wheel_speed and abs(carstate.wheel_velocities[2]-carstate.wheel_velocities[3])>berm_wheel_speed):
                gear = carstate.gear
                steering = -carstate.distance_from_center #(stuur ongeveer 1 de andere kant op: je zit links dus je stuurt rechts)
                brake = 0
                if speed < recover_speed_kmh*MPS_PER_KMH:
                    accelerate = recover_speed_kmh
                else:
                    accelerate = recover_speed_kmh
            elif speed > recover_speed and (abs(carstate.distance_from_center) < max_dist_centre or abs(carstate.distance_from_center)>berm_dist_centre or (abs(carstate.wheel_velocities[0]-carstate.wheel_velocities[1])<berm_wheel_speed and abs(carstate.wheel_velocities[2]-carstate.wheel_velocities[3])<berm_wheel_speed)):
                gear = carstate.gear # in de conditie hier boven gewijizgd dat de wielen aan beide zijden geen grootverschil meer hebben
                steering = 0
                brake = 0 # ik wil niet dat je remt
                accelerate = recover_speed_kmh*MPS_PER_KMH
            else:
                gear = carstate.gear
                bermCounter = 0
                steering = 0
                brake = 0
                accelerate = speed
            self.commands(carstate, brake, accelerate, steering, gear, command)

    def offTrack(self, carstate, speed, command, corner):
        global offtrackCounter
        global offtrack_time
        global recover_speed
        global max_dist_centre
        global off_track_constant
        global max_steer_degrees
        global recover

        recover = recover_time

        if offtrackCounter < offtrack_time:
            self.standardSteering(carstate, speed, command, corner)
        else:
            if abs(carstate.distance_from_center)>max_dist_centre:
                gear = carstate.gear
                steering = (carstate.angle - off_track_constant * carstate.distance_from_center)/max_steer_degrees
                brake = 0
                if speed < recover_speed_kmh*MPS_PER_KMH:
                    accelerate = recover_speed_kmh
                else:
                    accelerate = recover_speed_kmh
            elif speed > recover_speed and abs(carstate.distance_from_center) < max_dist_centre:
                gear = carstate.gear
                steering = 0
                #steering = (carstate.angle - off_track_constant * carstate.distance_from_center)/max_steer_degrees
                brake = 1
                accelerate = recover_speed_kmh
            else:
                gear = carstate.gear
                offtrackCounter = 0
                steering = 0
#               steering = (carstate.angle - off_track_constant * carstate.distance_from_center)/max_steer_degrees
                brake = 0
                accelerate = recover_speed_kmh
            self.commands(carstate, brake, accelerate, steering, gear, command)

    def adjustSteering(self, carstate, speed, command):
        global recover

        steering = self.avoidance(1, carstate.angle/max_steer_degrees, carstate)
        steering = self.steeringBounds(steering, speed)
        if carstate.gear == -1:
            gear = 1
        else:
            gear = carstate.gear
        if speed < recover_speed_kmh*MPS_PER_KMH or recover >0:
            accelerate = recover_speed_kmh
        else:
            accelerate = speed/MPS_PER_KMH
        brake = 0

        self.commands(carstate, brake, accelerate, steering, gear, command)

    def standardSteering(self, carstate, speed, command, corner):
        global max_steer_degrees
        global center_coef
        global recover
        global align_coef
        global center_coef
        global furthest_coef

        align_coef = 1
        center_coef = 11
        furthest_coef = 1
        sensor_degrees = [90, 75, 60, 45, 30, 20, 15, 10, 0, -10, -15, \
                            -20, -30, -45, -60, -75, -90] # Sign inversed to correspond to steering
        steeringDegrees = [1, 1, 1, 1, 0.8, 0.6, 0.45, 0.2, 0, \
				    -0.2, -0.45, -0.6, -0.8, -1, -1, -1, -1]
                    # 0, 5, 10, 15, 20, 30, 45, 60, 75, 90
        if(max(carstate.distances_from_edge) == -1) or carstate.angle > rho:
            steering = (carstate.angle - center_coef * carstate.distance_from_center)/max_steer_degrees
        else:
            degreesAlign = align_coef*carstate.angle
            degreesMiddle = center_coef*carstate.distance_from_center
            degreesFurthest = furthest_coef*sensor_degrees[np.argmax(np.array(carstate.distances_from_edge)[[0,1,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18]])]
            steering = degreesFurthest*(1-abs(carstate.distance_from_center)) + (degreesAlign-degreesMiddle)*abs(carstate.distance_from_center)

        steering = self.avoidance(0, steering, carstate)
        steering = self.steeringBounds(steering, speed)

        target_speed = self.speedNEAT(corner, carstate)

        speed_error = 1.0025 * target_speed * MPS_PER_KMH - speed
        AB = 2*(.5-1/(1+math.exp(speed_error)))
#        accelerate =  max(AB, 0)
        accelerate =  target_speed
        brake = -min(AB, 0)
        gear = carstate.gear
#        print("wat is in standard steer speederror " + repr(speed_error))
#        print("wat is AB? " + repr(AB))
        self.commands(carstate, brake, accelerate, steering, gear, command)

    def steeringBounds(self, steering, speed):
        global lastSteering
        global deviate_steer
        global max_speed
        steering = np.clip(steering, lastSteering - deviate_steer, lastSteering + deviate_steer)
        steering = np.clip(steering, -math.exp(-speed/speed_correction), math.exp(-speed/speed_correction))
        return steering

    def commands(self, carstate, brake, accelerate, steering, gear, command):
        global lastSteering
        global accelerationPrevious
        print("brake:" + repr(brake))
        print("accelerate:" + repr(accelerate))
        print("steering:" + repr(steering))
        print("gear:" + repr(gear) + " carstate: "+repr(carstate.gear) + " command: "+repr(command.gear) + " rpm "+repr(carstate.rpm))
        command.steering = self.steering_ctrl.control(
                steering,
                carstate.current_lap_time
        )
        lastSteering = steering
        command.brake = brake
        speed = math.sqrt(math.pow(carstate.speed_x,2)+math.pow(carstate.speed_y,2)+math.pow(carstate.speed_z,2))
        print("speed in kmh"  + repr(round(speed/MPS_PER_KMH)))
        speed_error = 1.0025 * accelerate * MPS_PER_KMH - speed
        AB = 2*(.5-1/(1+math.exp(speed_error)))
        # AB = accelerate + brake
        # speed_error = np.log(AB/(1-AB))
        # speed_accel_ms = 1.00025 * speed_error/3.6
        #
        print("accelerate " + repr(round(accelerate)))
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

#        print("wat is acceleration dan?" + repr(round(acceleration)))
#        command.accelerator = min(acceleration,1) # dit waseerst min(accelerate, 1)
        command.accelerator = max(AB,0) #

        print("en wat is hier de command.accelerator " + repr(command.accelerator) + " en de acceleration " + repr(acceleration))
        print("AB " + repr(AB))
        # stabilize use of gas and brake:
        # acceleration = math.pow(acceleration, 3)
        accelerationPrevious = command.accelerator
        if ((gear == 1 or gear == 2 or gear == 3) and carstate.rpm >= 9000):
            command.gear = gear + 1
#            print("deze 123" + repr(command.gear))
        elif ((gear == 4 or gear == 5) and carstate.rpm >= 8000):
            command.gear = gear + 1
#            print("deze 45")
        elif ((gear == 2 or gear == 3 or gear == 4) and carstate.rpm <= 3000):
            command.gear = gear - 1
#            print("deze dan? ")
        elif ((gear == 5 or gear == 6) and carstate.rpm <= 3500):
            command.gear = gear - 1
#            print("of deze dan")
        elif gear <= -1:
            command.gear = -1
        elif gear == 1:
            command.gear = 1
        elif not command.gear:
            command.gear = carstate.gear or 1
#        print("de echte command gear" + repr(command.gear))
        #print("gear:" + repr(gear) + " carstate: "+repr(carstate.gear) + " command: "+repr(command.gear) + " rpm "+repr(carstate.rpm))

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
        if((steerkind == 1 or abs(carstate.distance_from_center) > 0.8) and carstate.distance_from_center*steering>0): #if we are on the edge and intend to steer more in that direction, revert to plan
            steering = steerplan

    #    print("minsensor = " + repr(min_sensor))
    #    print("mindist = " + repr(min_range))
        return steering

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





    def speedNEAT(self, corner, carstate):
        # trackWidth = carstate.distances_from_edge[0]+carstate.distances_from_edge[18]
        # if(abs(corner) < 3 and max(carstate.distances_from_edge) < 90):
        #     v = 200 # Straight Approacing Corner
        #     print("Straight AC")
        # elif(abs(corner) < 3 ):
        #     v = 350 # Straight
        #     print("Straight")
        # elif(abs(corner) < 20):
        #     v = 200 # Full speed corner
        #     print("Full")
        # elif(abs(corner) < 35 and max(carstate.distances_from_edge) > trackWidth*5):
        #     v = 155 # Medium far distance
        #     print("MediumFar")
        # elif(abs(corner) < 35):
        #     v = 180 # Medium
        #     print("Medium")
        # elif(abs(corner) < 45):
        #     v = 150 # Slow
        #     print("Slow")
        # else:
        #     v = 79 # Hairpin
        #     print("Hairpin")
        # return v


        #
        global lastSteering
        global accelerationPrevious
        # ['ACCELERATION', 'STEERING', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0',
        # TRACK_EDGE_7', 'TRACK_EDGE_9', 'TRACK_EDGE_11', 'TRACK_EDGE_18', 'MAX_DISTANCE',
        # CORNER', 'TRACK_WIDTH', 'SPEED']]
        filenames = ['/home/student/Documents/torcs-server/torcs-client/NEAT100GensSlow',
                     '/home/student/Documents/torcs-server/torcs-client/NEAT100GensMedium',
                     '/home/student/Documents/torcs-server/torcs-client/NEAT100GensFull']
        if corner < 20:  # straight or full
            filename = filenames[0]
        elif corner < 35:  # medium
            filename = filenames[1]
        else:  # corner >= 35, slow or hairpin
            filename = filenames[2]

        inputList = [accelerationPrevious,
        lastSteering,
        carstate.distance_from_center,
        carstate.angle,
        carstate.distances_from_edge[0],
        carstate.distances_from_edge[7],
        carstate.distances_from_edge[9],
        carstate.distances_from_edge[11],
        carstate.distances_from_edge[18],
        max(carstate.distances_from_edge),
        self.cornerLearner(carstate),
        carstate.distances_from_edge[0] + carstate.distances_from_edge[18]]

        with gzip.open(filename) as f:
            data = pickle.load(f)
            # data[0] is NN
            # data[1] is ScalingFactor input
            # data[2] is speedFactor
        inputList = np.divide(np.asarray(inputList), np.asarray(data[1]))
        prediction = data[0].activate(inputList)[0]*data[2]
        print("Speedprediction: " + repr(prediction))
        return prediction
