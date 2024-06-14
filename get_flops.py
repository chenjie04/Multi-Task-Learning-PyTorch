"""
# python get_flops.py --config_env configs/env.yml --config_exp configs/pascal/squadnet/squadnet_base.yml --shape 512 512 # nudy 480 640
"""

from collections import defaultdict
import typing
import torch
import argparse

from utils.config import create_config
from utils.common_config import get_model, get_backbone
from termcolor import colored

from ptflops import get_model_complexity_info
# from fvcore.nn import FlopCountAnalysis

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--shape', nargs='+', type=int, 
                    help='The shape of input.')
args = parser.parse_args()

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'green'))

    # input
    shape = args.shape

    # print(colored('\nRetrieve backbone', 'blue'))
    # print("Complexity analysis on the backbone:")
    # # Get backbone
    # backbone = get_backbone(p)[0]
    # backbone.eval()

    # with torch.cuda.device(0):   
    #     macs, params = get_model_complexity_info(backbone, (3, *shape), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print(colored('\nRetrieve model', 'blue'))
    print("Complexity analysis on the whole model:")
    # Get model
    model = get_model(p)
    model.cuda()
    model.eval()
 

    with torch.cuda.device(0):   
        macs, params = get_model_complexity_info(model, (3, *shape), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # print(colored('\nComplexity analysis with fvcore:', 'blue'))
    # input = (torch.randn((1, 3, *shape)).cuda())
    # backbone = backbone.cuda()
    # backbone.eval()
    # flops = FlopCountAnalysis(backbone, input)
    # num_flops_backbone = flops.total()
    # model = model.cuda()
    # model.eval()
    # flops = FlopCountAnalysis(model, input)
    # num_flops_model = flops.total()
    # print("FLOPs in Backbone: %.2f " % (num_flops_backbone / 1e9))
    # print("FLOPs in Model: %.2f " % (num_flops_model / 1e9))

if __name__ == "__main__":
    main()
