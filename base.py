"""
All basic data structure in this file.
PLEASE DO NOT USE 'from .base import *' !!!
"""
import json


class NetworkUnit:
    def __init__(self, graph_part=[[]], cell_list=[]):
        self.graph_part = graph_part
        self.cell_list = cell_list
        self.name = ""
        self.scores = []
        return

    def load_from(self, path=""):
        with open(path, 'r') as f:
            network_dict = json.load(f)
            self.graph_part = network_dict['graph']
            self.name = network_dict['name']
        return

    def set_cell(self, new_cell=[]):
        self.cell_list[0] = new_cell
        return

class Dataset():
    def __init__(self):
        self.feature = None
        self.label = None
        self.shape = None
        return
        
    def load_from(self, path=""):
        # TODO
        return