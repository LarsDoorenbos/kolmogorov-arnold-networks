
import logging

import matplotlib.pyplot as plt

import torch
import ignite.distributed as idist

from kan.utils import expanduservars
from kan.trainer import Trainer, _build_datasets, _build_model, build_engine, load

LOGGER = logging.getLogger(__name__)


def visualize_layer(inputs, outputs, idx, output_path):
    print(f"Inputs shape {inputs.shape}")
    print(f"Outputs shape {outputs.shape}")

    fig, axs = plt.subplots(outputs.shape[1], outputs.shape[2], squeeze=False, figsize=(outputs.shape[2] * 5, outputs.shape[1] * 5), sharex=True, sharey=True)

    for i in range(outputs.shape[1]):
        for j in range(outputs.shape[2]):
            axs[i, j].scatter(inputs[:, i], outputs[:, i, j])
            axs[i, j].set_title(f"{i}, {j}")

    plt.tight_layout()
    plt.savefig(f"{output_path}/layer_{idx}.png")


def visualize(local_rank: int, params: dict):
    num_gpus = torch.cuda.device_count()
    LOGGER.info("%d GPUs available", num_gpus)
    output_path = expanduservars(params['output_path'])

    train_loader, validation_loader, test_loader = _build_datasets(params)

    input_shapes = [i.shape for i in train_loader.dataset[0] if hasattr(i, 'shape')]
    LOGGER.info("Input shapes: " + str(input_shapes))

    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        # Hack to find the number of intervals for the saved model
        params["num_intervals"] = torch.load(load_from)["model"]["layers.0.control_points"].shape[-1] - params["spline_order"]

    # Build the model, optimizer, trainer and training engine
    model = _build_model(params, input_shapes[0], input_shapes[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    trainer = Trainer(model, optimizer, params["lambda"], params["mu1"], params["mu2"])
    engine = build_engine(trainer, output_path, train_loader, validation_loader, params)

    if load_from is not None:
        load(load_from, trainer=trainer, engine=engine)

    model.eval()

    # Get all activations on test set
    inputs = {}
    activations = {}

    correct = 0
    total_mse = 0
    for idx, batch in enumerate(test_loader):
        x, y = batch
        x = x.to(idist.device())
        inps, acts = model.get_all_activations(x)

        if idx == 0:
            for layer, (i, a) in enumerate(zip(inps, acts)):
                inputs[layer] = i
                activations[layer] = a
        else:
            for layer, (i, a) in enumerate(zip(inps, acts)):
                inputs[layer] = torch.cat((inputs[layer], i))
                activations[layer] = torch.cat((activations[layer], a))

        y_pred = model(x)[0].cpu()
        correct += torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).item()

        mse = torch.nn.functional.mse_loss(y_pred, y, reduction='sum')
        total_mse += mse.item()

    print(f"Accuracy: {correct / len(test_loader.dataset)}")
    print(f"Mean MSE: {total_mse / len(test_loader.dataset)}")

    for i, a in inputs.items():
        print(f"Layer {i}")
        print(f"Activations shape {a.shape}")

    for i, a in activations.items():
        print(f"Layer {i}")
        print(f"Activations shape {a.shape}")

    for layer in range(len(model.layers)):
        visualize_layer(inputs[layer], activations[layer], layer, output_path)