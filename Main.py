#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22-01-04 ~ . 
# @Author  : KYS

import json
import argparse
import neptune
from trainer import Trainer

def main(config_file):
    
    params = json.load(open(config_file, 'rb'))
    params['experiment_name'] = params['experiment_name']
    
    # def neptune_init():
    #         print('\nLogging on ... ', end='')
    #         neptune.init()
    #         neptune.create_experiment(params={'learning_rate' : params["learning_rate"],
    #                                           'dual attention' : params["dual_attention"],
    #                                           'fgate encoder' : params["fgate_encoder"],
    #                                           'copy mechanism' : params["pointer"]
    #                                           })
        
    #neptune_init()
    
    print("Training models with params:")
    print(json.dumps(params, separators=("\n", ": "), indent=4))
    if params['mode'] == 'train': # train is contain valid
        trainer = Trainer(params)
        trainer.train()
    else :
        tester = Trainer(params)
        tester.test()

    #neptune.stop()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        help='configuration file for the training')         
    args = parser.parse_args()

    main(args.config_file)