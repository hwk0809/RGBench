from typing import Tuple, Dict, Optional, Union
from easydict import EasyDict
from omegaconf import OmegaConf, DictConfig, ListConfig, open_dict

from os import path as osp

def convert_dict(config: dict):
    new_dict = dict()
    for key, edict_item in config.items():
        if isinstance(edict_item, dict):
            new_dict[key] = convert_dict(config[key])
        else:
            if isinstance(edict_item, np.ndarray):
                new_dict[key] = edict_item.tolist()
            else:
                new_dict[key] = edict_item
    return new_dict


def config_completion(config: Union[Dict, EasyDict, DictConfig, str]) -> Union[DictConfig, ListConfig]:
    if isinstance(config, str):
        option = OmegaConf.load(config)
    elif isinstance(config, dict):
        config = convert_dict(config)
        option = OmegaConf.create(config)
    elif isinstance(config, EasyDict):
        config = convert_dict(dict(config))
        option = OmegaConf.create(config)
    elif isinstance(config, DictConfig):
        option = OmegaConf.create(config)
    else:
        raise NotImplementedError

    # TODO: check some result
    # with open_dict(option):
    #     # automatically override robot positions by reading calibration files
    #     option.compat.garment_type = GarmentTypeDef.from_string(option.compat.garment_type)
    #     with open(osp.join(option.compat.calibration_path, 'world_to_left_robot_transform.json'), 'r') as f:
    #         left_robot_to_world_transform = np.linalg.inv(np.array(json.load(f)))
    #     with open(osp.join(option.compat.calibration_path, 'world_to_right_robot_transform.json'), 'r') as f:
    #         right_robot_to_world_transform = np.linalg.inv(np.array(json.load(f)))
    #
    #     left_rpy_in_world = Rotation.from_matrix(left_robot_to_world_transform[:3, :3]).as_euler('xyz')
    #     right_rpy_in_world = Rotation.from_matrix(right_robot_to_world_transform[:3, :3]).as_euler('xyz')
    #
    #     # print("====================[ DEBUG ]====================")
    #     # print(option.planning.robot_init_positions, [tuple(left_robot_to_world_transform[:3,3].tolist()), tuple(right_robot_to_world_transform[:3,3].tolist())])
    #     # print(option.planning.robot_init_orientations, [tuple(left_rpy_in_world),tuple(right_rpy_in_world)])
    #     # print("====================[  END  ]====================")
    #     option.planning.robot_init_positions = [tuple(left_robot_to_world_transform[:3, 3].tolist()), tuple(right_robot_to_world_transform[:3, 3].tolist())]
    #     option.planning.robot_init_orientations = [tuple(left_rpy_in_world.tolist()), tuple(right_rpy_in_world.tolist())]

    return option
