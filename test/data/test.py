#!/home/bokanyie/.anaconda3/bin/python

import sys
import os

# make sure we're on repo root
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preparation import RawCSVtoMLN
from src.mln import MultiLayerNetwork

import json

# load config, read raw data and save compressed data
# config = json.load(open('src/config.json'))
# preparer = RawCSVtoMLN(**config)
# preparer.init_all()
# preparer.save_all()

# load into MLN from config
mln = MultiLayerNetwork(load_from_config=True, config_path='src/config.json')
print(mln.nodes.head(), mln.A)

