from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
# from pytocl.Opponents import State
import pandas as pd
import numpy as np
import math
import neat
import gzip
import pickle

stuckCounter = 0
steerPrevious = 0
accelerationPrevious = 0
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

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.
        """
        global recover
        command = Command()

        print("===================================================")
        print("recover " + repr(recover))

        corner = self.cornerLearner(carstate)
        print("corner: "+repr(corner) + " degrees")
        self.steer(carstate, 0.0, corner, command)
        v = self.speedNEAT(corner, carstate)
        if (abs(carstate.distance_from_center) > 1):
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
        steerplan = steering  # store our original steerplan

        for sensor in range(11, 23):  # kijk +60 tot -60 graden vooruit
            if (carstate.opponents[sensor] < min_range):
                min_sensor = sensor
            min_range = min(min_range, carstate.opponents[sensor])
        if (min_sensor == 12 and min_range / carstate.speed_x >= 0.30):  # closest opp 60+ degrees left and close enough
            steering -= 0.1  # steer a little to the right
        if (
                            min_sensor == 13 and min_range / carstate.speed_x >= 0.50 or min_sensor == 14 and min_range / carstate.speed_x >= 0.50):  # closest opp 50 or 40 degrees left and close enough
            steering -= 0.12
        if (min_sensor == 15 and min_range / carstate.speed_x >= 0.75):  # closest opp 30 degrees left and close enough
            steering -= 0.13
        if (min_sensor == 16 and min_range / carstate.speed_x >= 0.75):  # closest opp 20 degrees left and close enough
            steering -= 0.14
        if (min_sensor == 17 and min_range / carstate.speed_x >= 1):  # closest opp 10 degrees left and close enough
            steering -= 0.15

        if (min_sensor == 18 and min_range / carstate.speed_x >= 1):  # closest opp 10 degrees right and close enough
            steering += 0.15  # steer to the left
        if (min_sensor == 19 and min_range / carstate.speed_x >= 0.75):  # closest opp 20 degrees right and close enough
            steering += 0.14
        if (min_sensor == 20 and min_range / carstate.speed_x >= 0.75):  # closest opp 30 degrees right and close enough
            steering += 0.13
        if (
                            min_sensor == 21 and min_range / carstate.speed_x >= 0.50 or min_sensor == 22 and min_range / carstate.speed_x >= 0.50):  # closest opp 50 or 40 degrees right and close enough
            steering += 0.12
        if (
                        min_sensor == 23 and min_range / carstate.speed_x >= 0.30):  # closest opp 60+ degrees right and close enough
            steering -= 0.1  # steer a little to the right

        # above were the overtake scenarios ... these wlll be overwritten if any of the following are true
        if (min_sensor == 14 and min_range <= 10):  # closest opp 30+ degrees left and close
            steering -= firm  # steer to the right
        if (min_sensor == 15 and min_range <= 10):  # closest opp 20-30 degrees left and close
            steering -= firm  # steer to the right
        if (min_sensor == 16 and min_range <= 10):  # closest opp 10-20 degrees left and close
            steering -= firm  # steer to the right

        meeOfTegen = 0
        if (carstate.distance_from_center * steering > 0):
            meeOfTegen = -1
        else:
            meeOfTegen = 1

        if (
                            min_sensor == 17 and min_range <= 10 or min_sensor == 18 and min_range <= 10):  # closest opp dead ahead and close
            steering -= 0.1  # steer a little to the right
            print("collision imminent")
            if (steerkind == 1):
                if (steering < 0):
                    steering -= hard * meeOfTegen  # if we are on edge of track do not steer offTrack
                else:
                    steering += hard * meeOfTegen
            elif (steerkind == 0):  # ( steerkind == 0 is normalsteering)
                if (steering < 0):
                    steering -= hard
                else:
                    steering += hard
        if (min_sensor == 19 and min_range <= 10):  # closest opp 30+ degrees left and close
            steering += 0.25  # steer to the right
        if (min_sensor == 20 and min_range <= 10):  # closest opp 20-30 degrees left and close
            steering += 0.25  # steer to the right
        if (min_sensor == 21 and min_range <= 10):  # closest opp 10-20 degrees left and close
            steering += 0.25  # steer to the right

        # we will not go off track to avoid a collision. If the new steer will get us off track, we will revert to our original steer
        if (
                        steerkind == 1 and carstate.distance_from_center * steering > 0):  # if we are on the edge and intend to steer more in that direction, revert to plan
            steering = steerplan

        print("minsensor = " + repr(min_sensor))
        print("mindist = " + repr(min_range))
        return steering

    def steer(self, carstate, target_track_pos, corner, command):
        global closest
        global berm
        global stuckCounter
        global steerPrevious
        global wrongwaycounter
        global recover
        global unstuckCounter

        if (1 < abs(carstate.distance_from_center) < 1.1 and
                    abs(carstate.wheel_velocities[0] - carstate.wheel_velocities[1]) > 100 and
                    abs(carstate.wheel_velocities[2] - carstate.wheel_velocities[3]) > 100):
            berm += 1
            print("Berm" + repr(berm))
        else:
            berm = 0

        if (abs(carstate.angle) > 120 and carstate.gear >0):
            wrongwaycounter += 1
        else:
            wrongwaycounter = 0
        print("wrongwaycounter " + repr(wrongwaycounter))

        if (carstate.gear == -1 and carstate.distance_from_center * carstate.angle > 0):
            command.gear = 1
        print("stuckCounter " + repr(stuckCounter))
        speed = (carstate.speed_x ** 2 + carstate.speed_y ** 2 + carstate.speed_z ** 2) ** .5
        if (abs(carstate.angle) > 20 and
                    carstate.speed_x < 10 and
                    abs(carstate.distance_from_center) > 0.5) and (
                    carstate.distance_from_center * carstate.angle) < 0.0:
            stuckCounter = stuckCounter + 1
        else:
            stuckCounter = 0

        if (stuckCounter >= 200 or wrongwaycounter > 100):
            recover = 10000
            self.iAmStuck(carstate, target_track_pos, command)

        elif (berm > 100):
            recover = 1000
            self.solveberm(carstate, target_track_pos, command)
        elif (abs(carstate.distance_from_center) > 0.8):
            if (abs(carstate.distance_from_center) > 1):
                recover = 10000
                self.offTrack(carstate, target_track_pos, command)
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

    def adjustedSteering(self, carstate, target_track_pos, command):
        global steerPrevious
        global recover
        steering = (target_track_pos - carstate.distance_from_center) / 5
        if recover > 0:
            recover -= 1

        print("adjust steering ")
        print("steering:" + repr(steering))

        steerPrevious = steering

        if (carstate.speed_x > 0): steering = self.avoidance(1, steering, carstate)

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

    def accelerate(self, carstate, target_speed, command):
        global stuckCounter
        global recover
        global bermsolve
        global acceleration
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - math.sqrt(
            math.pow(carstate.speed_x, 2) + math.pow(carstate.speed_y, 2))
        print("speederror: ", repr(speed_error))
        AB = 2 * (.5 - 1 / (1 + math.exp(speed_error)))
        print("AB " + repr(AB))
        acceleration = self.acceleration_ctrl.control(
            AB,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)  # TODO

        if acceleration > 0:
            #if abs(carstate.distance_from_center) >= 1.0 or abs(carstate.distance_from_center) < 0.8 and abs(carstate.angle) > 30 or carstate.gear ==-1 or recover > 0:
            if(recover>0):
                acceleration = min(0.4, acceleration)
            if bermsolve == 1:
                    # acceleration = 0
                bermsolve = 0

            command.accelerator = min(acceleration, 1)
        if acceleration < 0:
            command.accelerator = 0
            command.brake = min(abs(acceleration),1) * 0.48

        accelerationPrevious = acceleration

        print("acceleration:" + repr(acceleration))

        if (carstate.gear == 1 or carstate.gear == 2 or carstate.gear == 3) and carstate.rpm >= 9000:
            command.gear = carstate.gear + 1
        elif (carstate.gear == 4 or carstate.gear == 5) and carstate.rpm >= 8000:
            command.gear = carstate.gear + 1
            # Shift down
        elif (carstate.gear == 2 or carstate.gear == 3 or carstate.gear == 4) and carstate.rpm <= 3000:
            command.gear = carstate.gear - 1
        elif (carstate.gear == 5 or carstate.gear == 6) and carstate.rpm <= 3500:
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
        # range_finder_angles are all angles -90, ..., 0, ..., 90
        #                -90-75-60-45-30, -20,  -15,  -10, -5,
        steeringDegrees = [1, 1, 1, 1, 1, 0.75, 0.63, 0.5, 0.25,
                           0, -0.25, -0.5, -0.63, -0.75, -1, -1, -1, -1, -1]
        #        			0,    5,  10,   15,  20, 30,45,60,75,90
        # Find the angle with farest distance closest to zero:
        maxIndices = [i for i, j in enumerate(carstate.distances_from_edge)
                      if j == max(carstate.distances_from_edge)]
        angles = [(i, self.range_finder_angles[i]) for i in maxIndices]
        angle = min(angles, key=lambda x: abs(x[1]))  # Tuple (index, angle)
        if (angle[1] >= 40):
            print("Maximum steering right")
            steering = -1;
        elif (angle[1] <= -40):
            print("Maximum steering left")
            steering = 1;
        else:
            # Steer is computed using distance measured by the track sensor with
            # maximum distance (and its two adjacent sensors)
            steering = steeringDegrees[angle[0]]
        if (abs(corner) > 35 and abs(corner) < 45):
            steering = steering * 2  # Slow
        elif (abs(corner) >= 45):
            steering = steering * 3  # Hairpin
        print("steering:" + repr(steering))
        steerPrevious = steering
        if recover > 0:
            recover -= 50
        if recover < 0:
            recover = 0

        if (carstate.speed_x > 0): steering = self.avoidance(0, steering, carstate)

        command.steering = self.steering_ctrl.control(
            steering,
            carstate.current_lap_time
        )

    def cornerLearner(self, carstate):
        N = len(carstate.distances_from_edge)
        border = np.zeros([N, 2])
        for i in range(N):
            border[i, :] = [-math.cos(i * math.pi / 18) * carstate.distances_from_edge[i],
                            math.sin(i * math.pi / 18) * carstate.distances_from_edge[i]]
        indexList = [i for i, j in enumerate(carstate.distances_from_edge) if j == max(carstate.distances_from_edge)]
        maxIndex = indexList[-1]
        minIndex = indexList[0]

        l = np.zeros([max(minIndex - 1, 0), 2])
        for i in range(minIndex - 1):
            dist = np.linalg.norm(border[i + 1, :] - border[i, :])
            l[i, :] = (border[i + 1, :] - border[i, :]) / dist

        r = np.zeros([max(N - maxIndex - 2, 0), 2])
        for i in range(N - maxIndex - 2):
            dist = np.linalg.norm(border[N - i - 2] - border[N - 1 - i])
            r[i, :] = (border[N - i - 2] - border[N - 1 - i]) / dist

        corner = 0
        for i in range(minIndex - 1 - 1):
            corner = corner + math.acos(np.dot(l[i, :], l[i + 1, :])) * 180 / math.pi * \
                              np.sign(l[i, 0] * l[i + 1, 1] - l[i, 1] * l[i + 1, 0])

        for i in range(N - 1 - maxIndex - 1 - 1):
            corner = corner + math.acos(np.dot(r[i, :], r[i + 1, :])) * 180 / math.pi * \
                              np.sign(r[i, 0] * r[i + 1, 1] - r[i, 1] * r[i + 1, 0])

        return corner

    # def speedCorner(self, corner, carstate):
    #     trackWidth = carstate.distances_from_edge[0] + carstate.distances_from_edge[18]
    #     if (abs(corner) < 3 and max(carstate.distances_from_edge) < 90):
    #         v = 200  # Straight Approacing Corner
    #         print("Straight AC")
    #     elif (abs(corner) < 3):
    #         v = 350  # Straight
    #         print("Straight")
    #     elif (abs(corner) < 20):
    #         v = 200  # Full speed corner
    #         print("Full")
    #     elif (abs(corner) < 35 and max(carstate.distances_from_edge) > trackWidth * 5):
    #         v = 155  # Medium far distance
    #         print("MediumFar")
    #     elif (abs(corner) < 35):
    #         v = 180  # Medium
    #         print("Medium")
    #     elif (abs(corner) < 45):
    #         v = 150  # Slow
    #         print("Slow")
    #     else:
    #         v = 79  # Hairpin
    #         print("Hairpin")
    #     return v
    def speedNEAT(self, corner, carstate):
        # ['ACCELERATION', 'STEERING', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0',
        # TRACK_EDGE_7', 'TRACK_EDGE_9', 'TRACK_EDGE_11', 'TRACK_EDGE_18', 'MAX_DISTANCE',
        # CORNER', 'TRACK_WIDTH', 'SPEED']]
        filenames = ['/home/student/Documents/offlineTraining/NEAT100GensSlow',
                     '/home/student/Documents/offlineTraining/NEAT100GensMedium',
                     '/home/student/Documents/offlineTraining/NEAT100GensFull']
        if corner < 20:  # straight or full
            filename = filenames[0]
        elif corner < 35:  # medium
            filename = filenames[1]
        else:  # corner <= 35, slow or hairpin
            filename = filenames[2]

        inputList = [accelerationPrevious,
        steerPrevious,
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
