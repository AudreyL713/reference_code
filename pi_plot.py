#!/usr/bin/python3
# -*- mode: python; coding: utf-8 -*-

#plotting imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import logging
import logging.config
#################

import argparse
import sys
import yaml

DEFAULT_CONFIG = {
    'all_grapher': {
        'file_location': "pact_scans/graph_scans",
        'start_dist': 0.0,
        'incr_dist': 1.0
        },
    'indiv_grapher': {
        'control_file': "pact_scans/graph_scans"
        },
    }

class All_Graph(object):
    def __init__(self, **kwargs):
        """Instance initialization.

        Args:
        """
        # Logger
        # self.__logger = logger
        # Grapher settings
        for key, value in DEFAULT_CONFIG['all_grapher'].items():
            if key in kwargs and kwargs[key]:
                setattr(self, key, kwargs[key])
                # try:
                #     x = getattr(self, key)
                #     print(x)
                # except AttributeError:
                #     print("GetAttr Failed")

                # try:
                #     y = self.key
                #     print(y)
                # except AttributeError:
                #     print("Access Failed")
            else:
                # self.__logger.debug("Using default beacon advertiser "
                #         f"configuration {key}: {value}.")
                setattr(self, key, value)
                print("Default attribute initialied")
                # try:
                #     x = getattr(self, key)
                #     print(x)
                # except AttributeError:
                #     print("GetAttr Failed")

                # try:
                #     y = self.key
                #     print(y)
                # except AttributeError:
                #     print("Access Failed")
        # self.__logger.info("Initialized beacon advertiser.")
        print("Initialized Grapher")

    def parse_data(self):
        # create dictionary of values to distances
        scans_dict = dict()

        curr_dist = getattr(self, "start_dist")
        print("Here1")
        # make a list of the valid csv files
        # must be saved as #.csv in order you want them to be graphed
        valid_files = list()
        for i in os.listdir(getattr(self, "file_location")):
            if (".csv" in i):
                valid_files.append(int(re.findall('\d+',i)[0]))
        print(valid_files)
        #loop through valid csv files
        for file in sorted(valid_files):
            file_name = getattr(self, "file_location") + "/" + str(file) + ".csv"

            #read RSSI column from file
            file_data = pd.read_csv(file_name)
            scan_values = file_data["RSSI"].tolist()
            
            #add list of RSSI values to dictionary and increment curr_dist
            scans_dict.update({curr_dist: scan_values})
            curr_dist = curr_dist + getattr(self, "incr_dist")
        print("scans initialized")
        return scans_dict

    def plot_all(self):
        scans_dict = parse_data(self)
        x_values = list(scans_dict.keys())
        
        fig, ax = plt.subplots()
        for x in x_values:
            ax.scatter([x] * len(scans_dict[x]), scans_dict[x], marker="o")

        scans_mean = np.array([np.mean(y) for x,y in sorted(scans_dict.items())])
        scans_std = np.array([np.std(y) for x,y in sorted(scans_dict.items())])
        ax.errorbar(x_values, scans_mean, yerr=scans_std, label="mean accuracy")
        
        ax.set_title("RSSI Values vs. Distance Between Pi's")
        ax.set_xlabel("Distance Between Pi's (inches)")
        ax.set_ylabel('RSSI Values')
        ax.grid(True)
        ax.legend()
        
        plt.show()
        print(scans_mean)

        pass

def setup_logger(config):
    """Setup and return logger based on configuration."""
    logging.config.dictConfig(config['config'])
    return logging.getLogger(config['name'])

def close_logger(logger):
    """Close logger."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

def load_config(parsed_args):
    """Load configuration.

    Loads beacon/scanner configuration from parsed input argument. Any
    expected keys not specified use values from default configuration.

    Args:
        parsed_args (Namespace): Parsed input arguments.

    Returns:
        Configuration dictionary.
    """
    # Load default configuration if none specified
    if parsed_args['config_yml'] is None:
        config = DEFAULT_CONFIG
    # Load configuration YAML
    else:
        with open(parsed_args['config_yml'], 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        config['all_grapher'] = {**DEFAULT_CONFIG['all_grapher'],
                **config['all_grapher']}
        config['indiv_grapher'] = {**DEFAULT_CONFIG['indiv_grapher'],
                **config['indiv_grapher']}
    # Merge configuration values with command line options
    for key, value in parsed_args.items():
        if value is not None:
            if key in config['all_grapher']:
                config['all_grapher'][key] = value
            if key in config['indiv_grapher']:
                config['indiv_grapher'][key] = value
    return config

def parse_args(args):
    """Input argument parser.

    Args:
        args (list): Input arguments as taken from sys.argv.

    Returns:
        Dictionary containing parsed input arguments. Keys are argument names.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=("Plotter for RSSI values in csv files. Command line "
                     "arguments will override their corresponding value in "
                     "a configuration file if specified."))
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-a', '--all_grapher', action='store_true',
                            help="Create line graph of data in files in specified folder")
    mode_group.add_argument('-i', '--indiv_grapher', action='store_true',
                            help="Create scatter plot of data in specified file")       
    parser.add_argument('--config_yml', help="Configuration YAML.")
    parser.add_argument('--file_location', help="Path to file")
    parser.add_argument('--start_dist', type=float,
            help="Distance between pi's for first reading")
    parser.add_argument('--incr_dist', type=float,
            help="Change in Distance between pi's from reading to reading")

    return vars(parser.parse_args(args))

def main(args):
    """Creates beacon and either starts advertising or scanning.

    Args:
        args (list): Arguments as provided by sys.argv.

    Returns:
        If advertising then no output (None) is returned. If scanning
        then scanned advertisements are returned in pandas.DataFrame.
    """
    # Initial setup
    
    parsed_args = parse_args(args)
    config = load_config(parsed_args)
    # logger = setup_logger(config['logger'])
    # logger.debug(f"Beacon configuration - {config['graph']}")

    try:
        if parsed_args['all_grapher']:
            # logger.info("Beacon advertiser mode selected.")
            grapher = All_Graph(**config['all_grapher'])
            grapher.plot_all()
        elif parsed_args['indiv_grapher']:
            print("This has not been implemented yet")
    except Exception:
        print(Exception)
    # finally:
    #     close_logger(logger)

if __name__ == "__main__":
    """Script execution."""
    main(sys.argv[1:])
