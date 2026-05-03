NUM_PARTITIONS = 100
NUM_SERVER_ROUNDS = 1#50

TARGET_DELTA = 1e-5 # probabilidade de falha da garantia de privacidade.
MAX_GRAD_NORM = 1.2 #Limiar de clipping C — cada gradiente por amostra é clipado para ter norma L2 ≤ C
NOISE_MULTIPLIER = 1.1  # ruído gaussiano
BATCH_SIZE = 32
EPOCHS = 1
FRACTION_FIT=0.3


LR = 0.1
MOMENTUM = 0.9

NUM_CLASSES = 10 # quantas classes de saída no dataset atual

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

MEAN = MNIST_MEAN
STD = MNIST_STD