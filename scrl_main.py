from scrl_trainer import SCRLTrainer, Watch, Config

def train(cfg: Config):
    with Watch('train.run'):
        trainer = SCRLTrainer.init_from_config(cfg=cfg)
        trainer.run()