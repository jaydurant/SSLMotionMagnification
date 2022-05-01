import os

import torch
import yaml
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base import ResNet50
import BYOLTrainer

print(torch.__version__)
torch.manual_seed(0)


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])


    # online network
    online_network = ResNet50(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

        # load pre-trained parameters
        load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                    map_location=torch.device(torch.device(device)))

        online_network.load_state_dict(load_params['online_network_state_dict'])


    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet50(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()