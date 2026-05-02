"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import train

NUM_SERVER_ROUNDS = 10


# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = NUM_SERVER_ROUNDS

    ndarrays = train.get_weights(train.Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


server_app = ServerApp(server_fn=server_fn)