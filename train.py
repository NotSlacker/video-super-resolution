import hydra
import lightning as L

from datasets import VideoYuvDataModule
from modules import VsrYuvModule

L.seed_everything(42, workers=True)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config):
    experiment_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # setup model
    VsrModel = hydra.utils.instantiate(config.model.type)

    n_frames = (
        len(config.dataset.sequence_lr_indices)
        - len(config.dataset.sequence_hr_indices)
        + 1
    )
    n_channels = config.dataset.n_channels
    scale_factor = config.dataset.scale_factor

    model = VsrModel(n_frames, n_channels, scale_factor, **config.model.parameters)

    # setup datamodule
    vfdm = VideoYuvDataModule(**config.dataloader, **config.dataset)

    # setup vsrmodule
    vsrm = VsrYuvModule(
        model,
        loss_params=config.loss,
        metric_params=config.metric,
        optimizer_params=config.optimizer,
    )

    # setup logger
    logger = L.pytorch.loggers.TensorBoardLogger(save_dir=experiment_dir, name="logs")

    # setup trainer
    trainer = L.Trainer(
        **config.trainer,
        logger=logger,
    )

    trainer.fit(vsrm, vfdm)

    # TODO implement multi-frame output test/predict
    # trainer.test(vsrm, vfdm, ckpt_path="best")


if __name__ == "__main__":
    main()
