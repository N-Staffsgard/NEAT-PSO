from torchvision import datasets, transforms
import torch
import neat
NO_DATA_POINTS = 1000
MNIST_CLASSES = 10
NO_WORKERS = 8
GENERATIONS = 100

# TODO: These are a beginning of an own implemenation of NEAT, not used anywhere yet.
class Genome:
    def __init__(self, key):
        self.key = key

class NEAT:
    def __init__(self, inputNodes, outputNodes):
        self.inputLayer =  torch.nn.Linear(in_features= inputNodes,out_features=1)
        self.relu1 = torch.nn.ReLU()
        self.outputLayer = torch.nn.Linear(1,outputNodes)
        self.model = [self.inputLayer, self.relu1,self.outputLayer]
        self.nLayers = 1


class Model:
    def __init__(self,inputNodes,outputNodes):
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform);
trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True);

train_dataset = dataset.data
train_dataset = train_dataset.view(-1,784)[:NO_DATA_POINTS]
train_labels = dataset.targets
train_labels = torch.eye(MNIST_CLASSES)[train_labels[:NO_DATA_POINTS]]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # print(f"Genome ID: {genome_id} : {genome}")
        fitness = []
        for xi, xo in zip(train_dataset, train_labels):
            # print(xi.shape)
            output = torch.tensor(net.activate(xi))
            loss = ((output-xo)**2).sum()
            fitness.append(-loss.item())
            # print(genome.fitness)
        genome.fitness = sum(fitness) / len(fitness)
        
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # print(f"Genome ID: {genome_id} : {genome}")
    fitness = []
    for xi, xo in zip(train_dataset, train_labels):
        # print(xi.shape)
        output = torch.tensor(net.activate(xi))
        loss = ((output-xo)**2).sum()
        fitness.append(-loss.item())
        # print(genome.fitness)
    return sum(fitness) / len(fitness)
        
"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()


    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    #import checkpoint if there exists any
    print("Checking if there exists any checkpoints")
    import glob
    import re
    # Get a list of all 'neat-checkpoint-xxx' files in the current directory
    files = glob.glob('neat-checkpoint-*')
    # Use a regular expression to extract the number from each filename
    numbers = [int(re.search(r'neat-checkpoint-(\d+)', file).group(1)) for file in files]
    # If there are any numbers, select the largest one
    if numbers:
        max_number = max(numbers)
        checkpoint_file = f'neat-checkpoint-{max_number}'
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        print(f"Checkpoint found: {checkpoint_file}")
    else:
        print("Checkpoint not found, running NEAT with new population.")

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    # Run for up to 300 generations.
    parallel_eval = ParallelEvaluator(NO_WORKERS,eval_genome)
    winner = p.run(parallel_eval.evaluate, GENERATIONS)
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    #Evaluation
    print("Results: ")
    print("==================================")

    acc_counter = []
    for xi, xo in zip(train_dataset, train_labels):
        output = torch.tensor(winner_net.activate(xi))
        print("target output {!r}, actual output {!r}".format(xo, output))
        print("target output {!r}, actual output {!r}".format(xo, torch.nn.functional.softmax(output, dim=0)))
        print(f"target output {torch.argmax(xo)}, actual output {torch.argmax(output)}")
        print("==================================")
        acc_counter.append(1.0 if torch.argmax(xo) ==torch.argmax(output) else 0.0)
    print(f"Overall Accuracy: {torch.tensor(acc_counter).mean().item()}")
    return winner_net, winner;



winner_1, winner_genome = run('NEAT/config-feedforward');
