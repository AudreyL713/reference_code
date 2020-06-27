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
# added imports
import threading
################
from bluetooth.ble import BeaconService
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
import sys
import time
from uuid import uuid1
import yaml

# Default configuration
LOG_NAME = 'pi_pact.log'
DEFAULT_CONFIG = {
    'advertiser': {
        'control_file': "advertiser_control",
        'timeout': None,
        'uuid': '',
        'major': 1,
        'minor': 1,
        'tx_power': 1,
        'interval': 200
        },
    'scanner': {
        'control_file': "scanner_control",
        'scan_prefix': "scan",
        'curr_file_id': 0,
        'timeout': None,
        'revisit': 1,
        'filters': {}
        },
    'logger': {
        'name': LOG_NAME,
        'config': {
            'version': 1,
            'formatters': {
                'full': {
                    'format': '%(asctime)s   %(module)-10s   %(levelname)-8s   %(message)s'},
                'brief': {
                    'format': '%(asctime)s   %(levelname)-8s   %(message)s'},
                },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'brief'
                    },
                'file': {
                    'class': 'logging.handlers.TimedRotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'full',
                    'filename': LOG_NAME,
                    'when': 'H',
                    'interval': 1
                    }
                },
            'loggers': {
                LOG_NAME: {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file']
                    }
                }
            }
        }
    }

# Universal settings
BLE_DEVICE = "hci0"
CONTROL_INTERVAL = 1 # (s)
MAX_TIMEOUT = 600 # (s)
ID_FILTERS = ['ADDRESS', 'UUID', 'MAJOR', 'MINOR', 'TX POWER']
MEASUREMENT_FILTERS = ['TIMESTAMP', 'RSSI']

# Limits
MAJOR_LIMITS = [1, 65535]
MINOR_LIMITS = [1, 65535]
TX_POWER_LIMITS = [-40, 4]
INTERVAL_LIMITS = [20, 10000] # (ms)
ALLOWABLE_FILTERS = ID_FILTERS+MEASUREMENT_FILTERS

def parse_data(file_location="pact_scans", start_dist=0.0, incr_dist=0.5):
    # create dictionary of values to distances
    scans_dict = dict()

    curr_dist = start_dist
    
    # make a list of the valid csv files
    # must be saved as #.csv in order you want them to be graphed
    valid_files = list()
    for i in os.listdir(file_location):
        if (".csv" in i):
            valid_files.append(int(re.findall('\d+',i)[0]))
    
    #loop through valid csv files
    for file in sorted(valid_files):
        file_name = str(file) + ".csv"

        #read RSSI column from file
        file_data = pd.read_csv(file_name)
        scan_values = file_data["RSSI"].tolist()
        
        #add list of RSSI values to dictionary and increment curr_dist
        scans_dict.update({curr_dist: scan_values})
        curr_dist = curr_dist + incr_dist
    
    return scans_dict

def plot_all(file_location="pact_scans", start_dist=0.0, incr_dist=0.5):
    scans_dict = parse_data(file_location=file_location, start_dist=start_dist, incr_dist=incr_dist)
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
        config['advertiser'] = {**DEFAULT_CONFIG['advertiser'],
                **config['advertiser']}
        config['scanner'] = {**DEFAULT_CONFIG['scanner'],
                **config['scanner']}
    # Merge configuration values with command line options
    for key, value in parsed_args.items():
        if value is not None:
            if key in config['advertiser']:
                config['advertiser'][key] = value
            if key in config['scanner']:
                config['scanner'][key] = value
    # Remove malformed filters
    if config['scanner']['filters'] is not None:
        filters_to_remove = []
        for key, value in config['scanner']['filters'].items():
            if key not in ALLOWABLE_FILTERS or not isinstance(value, list):
                filters_to_remove.append(key)
            elif key in MEASUREMENT_FILTERS and len(value) != 2:
                filters_to_remove.append(key)
        for filter_to_remove in filters_to_remove:
            del config['scanner']['filters'][filter_to_remove]
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
        description=("BLE beacon advertiser or scanner. Command line "
                     "arguments will override their corresponding value in "
                     "a configuration file if specified."))
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-g', '--graph', action='store_true',
                            help="Graph Data")
    # parser.add_argument('--config_yml', help="Configuration YAML.")
    return vars(parser.parse_args(args))

def main(args):
    """Creates beacon and either starts advertising or scanning.

    Args:
        args (list): Arguments as provided by sys.argv.

    Returns:
        If advertising then no output (None) is returned. If scanning
        then scanned advertisements are returned in pandas.DataFrame.
    """
    # # Initial setup
    
    # parsed_args = parse_args(args)
    # config = load_config(parsed_args)
    # logger = setup_logger(config['logger'])
    # logger.debug(f"Beacon configuration - {config['graph']}")


    # # Create and start beacon advertiser or scanner
    # try:
    #     if parsed_args['advertiser']:
    #         logger.info("Beacon advertiser mode selected.")
    #         advertiser = Advertiser(logger, **config['advertiser'])
    #         advertiser.advertise()
    #         output = None

    # except Exception:
    #     logger.exception("Fatal exception encountered")
    # finally:
    #     close_logger(logger)
    plot_all()

if __name__ == "__main__":
    """Script execution."""
    main(sys.argv[1:])
