import random
import neat
import gzip
import pickle

Data = pickle.load(open("trainDataNEAT.p", "rb"))
trainDataSet = Data[0]
observedDataSet = Data[1]
filenames = Data[2]
scalingFactor = Data[3]
speedFactor = Data[4]

# Run until a solution is found or maximum generations is reached
maximumGens = 150
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
                genome.fitness -= (prediction[0] - observedData[i]) ** 2 / sampleSize * speedFactor

    winner = p.run(eval_genomes, maximumGens)

    # Display the winning genome.
    print(winner)

    # Show output of the most fit genome against training data.
    minLines = random.sample(range(len(trainData)), 1)[0]  # Between 0 and max Lines
    maxLines = minLines + 10  # Between min Lines and 32400
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    for input, observed in zip(trainData[minLines:maxLines], observedData[minLines:maxLines]):
        prediction = net.activate(input)
        print("Pred: " + repr(speedFactor * prediction[0]) + " Observ: " + repr(speedFactor * observed))

    # Save genome
    net = neat.nn.FeedForwardNetwork.create(p.best_genome, config)
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump((net, scalingFactor.tolist(), speedFactor), f, protocol=pickle.HIGHEST_PROTOCOL)

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
