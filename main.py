"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

from flwr.simulation import run_simulation
from server import ServerApp, server_fn
from client import ClientApp, client_fn

NUM_PARTITIONS = 100
NUM_SERVER_ROUNDS = 10

TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 2.0

server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)


hist = run_simulation(
    server_app=server_app, client_app=client_app, num_supernodes=NUM_PARTITIONS
)
