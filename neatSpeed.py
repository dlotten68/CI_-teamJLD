import pandas as pd
import numpy as np
import math
import random
import neat
import gzip
import pickle
from __future__ import print_function

aalborg = pd.read_csv('aalborg.csv', index_col=None, engine='python')
alpine1 = pd.read_csv('alpine-1.csv', index_col=None, engine='python')
fspeedway = pd.read_csv('f-speedway.csv', index_col=None, engine='python')
testdrive = pd.read_csv('testdrive.csv', index_col=0, engine='python')

#  Drop BRAKE, no brake in testdrive
aalborg.drop(['BRAKE'], axis=1, inplace=True)
alpine1.drop(['BRAKE'], axis=1, inplace=True)
fspeedway.drop(['BRAKE'], axis=1, inplace=True)

#  Calculate speed with euclidean distance
testdrive['SPEED'] = (testdrive['speedX'] ** 2 + testdrive['speedY'] ** 2) ** .5
#  Drop columns not in aalborg, alpine1, fspeedway
testdrive.drop(['speedX', 'speedY', 'speedZ', 'rpm', 'gear', 'setGear'], axis=1, inplace=True)
#  Rearrange columns of testdrive
testdrive = testdrive[
    ['setAcceleration', 'setSteer', 'SPEED', 'trackPos', 'angle', 'track0', 'track1', 'track2', 'track3', 'track4',
     'track5', 'track6', 'track7', 'track8', 'track9', 'track10', 'track11', 'track12', 'track13', 'track14', 'track15',
     'track16', 'track17', 'track18']]
#  Set new names testdrive, so they can be concate
testdrive.columns = [
    ['ACCELERATION', 'STEERING', 'SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1',
     'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8',
     'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14',
     'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']]
#  Concatenate dataframes to one super train data frame
trainDataFrame = pd.concat([aalborg, alpine1, fspeedway, testdrive])
#  Make a list of tracknames
trackNAMES = ['TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5',
              'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11',
              'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17',
              'TRACK_EDGE_18']
#  Drop every row with an NA
trainDataFrame.dropna(axis=0, inplace=True)

trainDataFrame['MAX_DISTANCE'] = trainDataFrame[trackNAMES].max(axis=1)

def cornerLearner(trackData):
    N = len(trackData)
    border = np.zeros([N, 2])
    for i in range(N):
        border[i, :] = [-math.cos(i * math.pi / 18) * trackData[i],
                        math.sin(i * math.pi / 18) * trackData[i]]
    indexList = [i for i, j in enumerate(trackData) if j == max(trackData)]
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
        corner = corner + math.acos(np.clip(np.dot(l[i, :], l[i + 1, :]), -1, 1)) * 180 / math.pi * \
                          np.sign(l[i, 0] * l[i + 1, 1] - l[i, 1] * l[i + 1, 0])
    for i in range(N - 1 - maxIndex - 1 - 1):
        corner = corner + math.acos(np.clip(np.dot(r[i, :], r[i + 1, :]), -1, 1)) * 180 / math.pi * \
                          np.sign(r[i, 0] * r[i + 1, 1] - r[i, 1] * r[i + 1, 0])
    return corner

trainDataFrame['CORNER'] = trainDataFrame[trackNAMES].apply(func=cornerLearner, axis=1);
trainDataFrame['TRACK_WIDTH'] = trainDataFrame['TRACK_EDGE_0'] +  trainDataFrame['TRACK_EDGE_18']
trainDataFrame = trainDataFrame[['ACCELERATION', 'STEERING', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0',
                                 'TRACK_EDGE_7', 'TRACK_EDGE_9', 'TRACK_EDGE_11', 'TRACK_EDGE_18', 'MAX_DISTANCE',
                                 'CORNER', 'TRACK_WIDTH', 'SPEED']]

scalingFactor = trainDataFrame.drop(['SPEED'], axis=1).abs().max(axis=0)
speedFactor = trainDataFrame['SPEED'].abs().max()

trainDataSlow = trainDataFrame.loc[trainDataFrame['CORNER'] < 20 ]
trainDataMedium = trainDataFrame.loc[((trainDataFrame['CORNER'] >= 20) & (trainDataFrame['CORNER'] < 35))]
trainDataFull = trainDataFrame.loc[trainDataFrame['CORNER'] >= 35]
speedSlow = trainDataSlow['SPEED']/speedFactor
speedMedium = trainDataMedium['SPEED']/speedFactor
speedFull = trainDataFull['SPEED']/speedFactor

trainDataSlow = trainDataSlow.drop(['SPEED'], axis=1)
trainDataMedium = trainDataMedium.drop(['SPEED'], axis=1)
trainDataFull = trainDataFull.drop(['SPEED'], axis=1)

trainDataSlow = trainDataSlow/scalingFactor
trainDataMedium = trainDataMedium/scalingFactor
trainDataFull = trainDataFull/scalingFactor

trainDataSlow = trainDataSlow.apply(tuple, axis=1).tolist()
trainDataMedium = trainDataMedium.apply(tuple, axis=1).tolist()
trainDataFull = trainDataFull.apply(tuple, axis=1).tolist()
observedDataSlow = speedSlow.values
observedDataMedium = speedMedium.values
observedDataFull = speedFull.values

trainDataSet = [trainDataSlow, trainDataMedium, trainDataFull]
observedDataSet = [observedDataSlow, observedDataMedium, observedDataFull]
filenames = ['NEAT100GensSlow', 'NEAT100GensMedium', 'NEAT100GensFull']

# Run until a solution is found or maximum generations is reached
maximumGens = 100
sampleSize = 500

for i in range(len(filenames)):
    trainData = trainDataSet[i]
    observedData = observedDataSet[i]
    filename = filenames[i]

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'configuration')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0.0  # set our maximum fitnes
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # Zip input (list of tuples every tuple an input row) with output (tuples with 1 value)
            randomIndices = random.sample(range(len(trainData)), sampleSize)
            for i in randomIndices:
                prediction = net.activate(trainData[i])
                # Error function, difference predicted and real output
                genome.fitness -= (prediction[0] - observedData[i]) ** 2/sampleSize*speedFactor

    winner = p.run(eval_genomes, maximumGens)

    # Display the winning genome.
    print(winner)

    # Show output of the most fit genome against training data.
    minLines = random.sample(range(len(trainData)), 1)[0]  # Between 0 and max Lines
    maxLines = minLines + 10  # Between min Lines and 32400
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    for input, observed in zip(trainData[minLines:maxLines], observedData[minLines:maxLines]):
        prediction = net.activate(input)
        print("Pred: " + repr(speedFactor*prediction[0]) + " Observ: " + repr(speedFactor*observed))

    # Save genome
    net = neat.nn.FeedForwardNetwork.create(p.best_genome, config)
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump((net, scalingFactor, speedFactor), f, protocol=pickle.HIGHEST_PROTOCOL)

########## BELOW IS LOAD CODE ##########
#
# with gzip.open(filename) as f:
#     data = pickle.load(f)
# prediction = data[0].activate(trainData[5])
#
# ########## TEST FOR MYSELF ########
# minLines = 2000  # Between 0 and max Lines
# maxLines = 2010  # Between min Lines and 32400
# for input, observed in zip(trainData[minLines:maxLines], observedData[minLines:maxLines]):
#     prediction = data[0].activate(input)
#     print("Pred: " + repr(data[2]*prediction[0]) + " Observ: " + repr(data[2]*observed))
