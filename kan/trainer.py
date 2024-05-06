
from dataclasses import dataclass
import logging
import importlib
import os
from typing import Iterable, Optional, Tuple, Dict, Union, Any, cast

import matplotlib.pyplot as plt

from torch import nn
import torch

from ignite.engine import Engine, Events
import ignite.distributed as idist
from ignite.utils import setup_logger
from ignite.metrics import Frequency, MeanSquaredError, RootMeanSquaredError
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import WandBLogger
from ignite.contrib.metrics import GpuInfo

from .builder import KAN, build_model
from .utils import archive_code, expanduservars

LOGGER = logging.getLogger(__name__)
Model = Union[KAN, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _flatten(m: Model) -> KAN:
    if isinstance(m, KAN):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return cast(KAN, m.module)
    else:
        raise TypeError("type(m) should be one of (KAN, DataParallel, DistributedDataParallel)")


@dataclass
class Trainer:

    model: Model
    optimizer: torch.optim.Optimizer

    @property
    def flat_model(self):
        """View of the model without DataParallel wrappers."""
        return _flatten(self.model)

    def train_step(self, _: Engine, batch) -> dict:
        self.model.train()

        x, y = batch
        batch_size = x.shape[0]
        
        device = idist.device()
        x = x.to(device, non_blocking=True)

        output = self.model(x)
        
        # Penalize the difference between real and estimated noise
        loss = nn.functional.mse_loss(output, y.to(device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"num_items": batch_size, "loss": loss.item()}

    @torch.no_grad()
    def test_step(self, _: Engine, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.model.eval()

        x, y = batch
        
        device = idist.device()
        x = x.to(device, non_blocking=True)

        output = self.model(x)
        
        return (output, y.to(device))

    def objects_to_save(self, engine: Optional[Engine] = None) -> dict:
        to_save: Dict[str, Any] = {
            "model": self.flat_model.unet,
            "optimizer": self.optimizer
        }

        if engine is not None:
            to_save["engine"] = engine

        return to_save


def build_engine(trainer: Trainer, output_path: str, train_loader: Iterable, validation_loader: Iterable, params: Dict) -> Engine:

    engine = Engine(trainer.train_step)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")

    engine_val = Engine(trainer.test_step)
    mse = MeanSquaredError()
    rmse = RootMeanSquaredError()
    mse.attach(engine_val, "mse")
    rmse.attach(engine_val, "rmse")

    if params["wandb"]:
        tb_logger = WandBLogger(project='KAN', config=params)

        tb_logger.attach_output_handler(
            engine,
            Events.ITERATION_COMPLETED(every=params["log_every"]),
            tag="training",
            output_transform=lambda x: x,
            metric_names=["imgs/s"],
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )

        tb_logger.attach_output_handler(
            engine_val,
            Events.EPOCH_COMPLETED,
            tag="testing",
            metric_names=["mse", 'rmse'],
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )
    
    # Display some info every 100 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=50))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_info(engine: Engine):
        LOGGER.info(
            "epoch=%d, iter=%d, speed=%.2fimg/s, loss=%.4g, gpu:0 util=%.2f%%",
            engine.state.epoch,
            engine.state.iteration,
            engine.state.metrics["imgs/s"],
            engine.state.output["loss"],
            engine.state.metrics["gpu:0 util(%)"]
        )

    # Update grid regularly. Double the grid resolution every refine_grid_every epochs
    @torch.no_grad()
    @engine.on(Events.EPOCH_STARTED(every=5))
    def update_grid(_: Engine):
        refine = engine.state.epoch % params["refine_grid_every"] == 0
        update = engine.state.epoch <= params["stop_updating_grid_after"]

        if update and refine:
            LOGGER.info("Updating and refining grids...")
        elif update:
            LOGGER.info("Updating grids...")
        elif refine:
            LOGGER.info("Refining grids...")
        else:
            return
        
        trainer.flat_model.train()
        batch = next(iter(train_loader))[0].to(idist.device())
        trainer.flat_model.update_grids(batch, update=update, refine=refine)

        # Need to re-initialize optimizer if grids are refined, as dimensions of control points change
        if refine:
            LOGGER.info("Re-initializing optimizer...")
            trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=params["learning_rate"])
        
    # Compute the validation score every so often
    @engine.on(Events.ITERATION_COMPLETED(every=params["evaluate_every"]))
    def compute_fid(_: Engine):
        LOGGER.info("Validation MSE computation...")
        engine_val.run(validation_loader, max_epochs=1)
        LOGGER.info("Validation MSE score: %.4g, RMSE score: %.4g", engine_val.state.metrics["mse"], engine_val.state.metrics["rmse"])

    # Visualize something every so often
    @engine.on(Events.ITERATION_COMPLETED(every=100))
    def visualize(_: Engine):
        LOGGER.info("Visualizing...")
        x, y = next(iter(validation_loader))
        x = x.to(idist.device())
        y = y.to(idist.device())
        output = trainer.model(x).detach().cpu().numpy()[:, 0]

        x = x.cpu().numpy()[:, 0]
        y = y.cpu().numpy()[:, 0]

        plt.figure()
        plt.scatter(x, y, label='Real')
        plt.scatter(x, output, label='Estimated')

        plt.legend()
        plt.savefig(os.path.join(output_path, f"visualization_{engine.state.iteration}.png"))
        plt.close()

    return engine


def load(filename: str, trainer: Trainer, engine: Engine):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine)
    ModelCheckpoint.load_objects(to_load, state)


def _build_model(params: dict, input_shape, output_shape) -> Model:
    model: Model = build_model(
        params,
        input_shape,
        output_shape
    ).to(idist.device())

    return model


def _build_datasets(params: dict):
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)

    train_dataset = dataset_module.training_dataset()  # type: ignore
    validation_dataset = dataset_module.validation_dataset()  # type: ignore
    test_dataset = dataset_module.test_dataset()  # type: ignore

    LOGGER.info("%d samples in dataset '%s'", len(train_dataset), dataset_file)
    LOGGER.info("%d samples in validation dataset '%s'", len(validation_dataset), dataset_file)
    LOGGER.info("%d samples in test dataset '%s'", len(test_dataset), dataset_file)

    batch_size = params['batch_size'] 

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=params["mp_loaders"])
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=params["mp_loaders"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=params["mp_loaders"])

    return train_loader, validation_loader, test_loader


def train_kan(local_rank: int, params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    archive_code(output_path)

    num_gpus = torch.cuda.device_count()
    LOGGER.info("%d GPUs available", num_gpus)

    train_loader, validation_loader, test_loader = _build_datasets(params)

    input_shapes = [i.shape for i in train_loader.dataset[0] if hasattr(i, 'shape')]
    LOGGER.info("Input shapes: " + str(input_shapes))

    # Build the model, optimizer, trainer and training engine
    model = _build_model(params, input_shapes[0], input_shapes[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    trainer = Trainer(model, optimizer)
    engine = build_engine(trainer, output_path, train_loader, validation_loader, params)

    # Load a model (if requested in params.yml) to continue training from it
    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        load(load_from, trainer=trainer, engine=engine)

    # Run the training engine for the requested number of epochs
    engine.run(train_loader, max_epochs=params["max_epochs"])