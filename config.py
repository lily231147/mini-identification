"""
written by lily
email: lily231147@gmail.com
"""
from typing import Literal, Union
from pathlib import Path
import json


class Config:
    """ config """
    # optimizer and scheduler
    optimizer: Literal["adam", "adamw", "sgd"] = 'adam'
    optimizer_args = {
        'adam': {'betas': [0.9, 0.99]},
        'adamw': {'weight_decay': 0.},
        'sgd': {'momentum': None}}
    scheduler = {'step_size': 40, 'gamma': 0.1}
    # appliance
    app_ids = {'redd': {'furnace': 10, 'washer-dryer': 14, 'microwave': 16,},
               'ukdale': {'kettle': 8, 'rice-cooker': 9, 'dishwasher': 13, 'microwave': 15}}
    n_class = {'redd': {'furnace': 6, 'washer-dryer': 4, 'microwave': 4}, 
               'ukdale': {'kettle': 2, 'rice-cooker': 2, 'dishwasher': 4, 'microwave': 2}}
    partition = {'redd':{'furnace': [322, 374, 1392], 'washer-dryer': [690, 993, 1392], 'microwave': [684, 1104, 1392]},
                 'ukdale': {'kettle': [4768, 8676, 11880], 'rice-cooker': [6887, 9083, 11880], 
                            'dishwasher': [6556, 8581, 11880], 'microwave': [6712, 8812, 11880]}}
    amplitude_threshold = {
        'redd':{
            'furnace': {
                'max': [194.04000000000008, 485.74, 437.24, 1360.74, 148.33000000000004, 670.5600000000001],
                'min': [102.6099999999999, 353.02, 396.08, 1148.7200000000003, 107.90000000000009, 600.31]
            },
            'washer-dryer': {
                'max': [3204.4799999999996, 2963.92, 2487.9500000000003, 511.35],
                'min': [2845.33, 2317.26, 2042.58, 406.8399999999999]
            },
            'microwave': {
                'max': [1957.1799999999998, 1882.19, 165.42000000000002, 10000],
                'min': [1587.6400000000003, 1573.4299999999998, 100, 29.570000000000007]
            }
        },
        'ukdale':{
            'kettle': {'max': [3200, 3200], 'min': [2800, 2700]},
            'rice-cooker': {'max': [450, 400], 'min': [350, 340]},
            'dishwasher': {'max': [2186.4799999999996, 2030, 135, 129], 'min': [1913.33, 1828, 84, 93]},
            'microwave': {'max': [1400, 1450], 'min': [1100, 1100]}
        }
    }
    stable_threshold = {
        'redd':{
            'furnace': {
                'front': [194.04000000000008, 485.74, 437.24, 1360.74, 148.33000000000004, 670.5600000000001],
                'end': [102.6099999999999, 353.02, 396.08, 1148.7200000000003, 107.90000000000009, 600.31]
            },
            'washer-dryer': {
                'front': [3204.4799999999996, 2963.92, 2487.9500000000003, 511.35],
                'end': [2845.33, 2317.26, 2042.58, 406.8399999999999]
            },
            'microwave': {
                'front': [1957.1799999999998, 1882.19, 165.42000000000002, 54],
                'end': [1587.6400000000003, 1573.4299999999998, 100, 29.570000000000007]
            }
        },
        'ukdale':{
            'kettle': {'front': [3200, 3200], 'end': [2800, 2700]},
            'rice-cooker': {'front': [42.5, 25], 'end': [42.5, 25]},
            'dishwasher': {'front': [2186.4799999999996, 2030, 135, 129], 'end': [1913.33, 1828, 84, 93]},
            'microwave': {'front': [400, 400], 'end': [400, 400]}
        }
    }
import json

# 获取类的静态字段
class_fields = {key: value for key, value in Config.__dict__.items() if not key.startswith('__') and not callable(value)}

config = Config()
with open('metadata.json', 'w') as file:
    json.dump(class_fields, file, indent=4)