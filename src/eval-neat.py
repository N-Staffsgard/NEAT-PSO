from torchvision import datasets, transforms
import torch
import neat
import argparse
NO_PICTURES_EVALUATED = 100
parser = argparse.ArgumentParser(description="NEAT evaluator")
parser.add_argument('--checkpoint',type=str, required=True, help='The path to the checkpoint file')
args = parser.parse_args()

print(f"loading {args.checkpoint}")
p = neat.Checkpointer.restore_checkpoint(args.checkpoint)

config_path = "NEAT/config-feedforward"
print(f"loading config from {config_path}")
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

print("Loading MNIST data set")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform);
train_dataset = dataset.data
train_dataset = train_dataset.view(-1,784)[:NO_PICTURES_EVALUATED]
train_labels = dataset.targets
train_labels = torch.eye(10)[train_labels[:NO_PICTURES_EVALUATED]]

print("Building neural net")
best_genome_tuple = max(p.population.items(), key=lambda item: item[1].fitness if item[1].fitness is not None else -float('inf'))
best_genome = best_genome_tuple[1]
net = neat.nn.FeedForwardNetwork.create(best_genome,config)

print("printing results")
acc_counter = []
for i, (inputs, labels) in enumerate(zip(train_dataset, train_labels)):
    output = torch.tensor(net.activate(inputs))
    if i < 10 :    
        print(f"target output {labels}, actual output {output}")
        # print(f"target output {labels}, softmax output {torch.nn.functional.softmax(output, dim=0)}")
        print(f"target output {torch.argmax(labels)}, actual output {torch.argmax(output)}")
        print("==================================")
    acc_counter.append(1.0 if torch.argmax(labels) ==torch.argmax(output) else 0.0)
print(f"Overall Accuracy: {torch.tensor(acc_counter).mean().item()}")