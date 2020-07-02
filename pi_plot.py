#!/usr/bin/python3
# -*- mode: python; coding: utf-8 -*-

#plotting imports
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import logging
import logging.config
import statistics
#################

import argparse
import sys
import yaml

DEFAULT_CONFIG = {
    'all_grapher': {
        'file_location': "pact_scans/graph_scans",
        'scan_prefix': "scan_",
        'graph_title': "RSSI Values vs. Distance Between Pi's",
        'y_label': "RSSI Values",
        'x_label': "Distance Between Pi's (inches)",
        'best_fit': 1,
        'start_dist': 0.0,
        'incr_dist': 1.0
        },
    'indiv_plotter': {
        'file_location': "pact_scans/graph_scans/scan_0.csv"
        },
    }

BEST_FIT_LIMITS = [-1, 5]

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
            else:
                # self.__logger.debug("Using default beacon advertiser "
                #         f"configuration {key}: {value}.")
                setattr(self, key, value)
                print("Default attribute initialized")

        # self.__logger.info("Initialized beacon advertiser.")
        print("Initialized Grapher")

    @property
    def best_fit(self):
        """BLE beacon advertiser TX power value getter."""
        return self.__best_fit

    @best_fit.setter
    def best_fit(self, value):
        """BLE beacon Beacon advertiser TX power setter.

        Raises:
            TypeError: Beacon advertiser TX power must be an integer.
            ValueError: Beacon advertiser TX power must be in [-40, 4].
         """
        if not isinstance(value, int):
            raise TypeError("Degree of Best Fit Line must be an integer.")
        elif value < BEST_FIT_LIMITS[0] or value > BEST_FIT_LIMITS[1]:
            raise ValueError("Degree of Best Fit Line must be in range "
                    f"{BEST_FIT_LIMITS}.")
        self.__best_fit = value

    def parse_data(self):
        # create dictionary of values to distances
        scans_dict = dict()
        curr_dist = self.start_dist
        # make a list of the valid csv files
        # must be saved as #.csv in order you want them to be graphed
        valid_files = list()
        for i in os.listdir(self.file_location):
            if (".csv" in i):
                valid_files.append(int(re.findall('\d+',i)[0]))
        #loop through valid csv files
        for file in sorted(valid_files):
            file_name =  self.file_location + "/" + self.scan_prefix + str(file) + ".csv"

            #read RSSI column from file
            file_data = pd.read_csv(file_name)
            scan_values = file_data["RSSI"].tolist()
            
            #add list of RSSI values to dictionary and increment curr_dist
            scans_dict.update({curr_dist: scan_values})
            curr_dist = curr_dist + self.incr_dist
        return scans_dict

    def plot_all(self):
        scans_dict = self.parse_data()
        x_values = list(scans_dict.keys())
        
        fig, ax = plt.subplots()
        for x in x_values:
            ax.scatter([x] * len(scans_dict[x]), scans_dict[x], marker="o")

        scans_mean = np.array([np.mean(y) for x,y in sorted(scans_dict.items())])
        scans_std = np.array([np.std(y) for x,y in sorted(scans_dict.items())])
        ax.errorbar(x_values, scans_mean, yerr=scans_std, label="mean accuracy")
        
        ax.set_title(self.graph_title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.grid(True)

        if self.best_fit==1:
            x = np.array(x_values)
            m, b = np.polyfit(x_values, scans_mean, 1)
            equation = f"y = {round(m,4)}x + {round(b,4)}"
            ax.plot(x, m*x+b, '-r', label=equation)
        elif self.best_fit==2:
            x = np.linspace(x_values[0],x_values[-1],100)
            x1, m, b = np.polyfit(x_values, scans_mean, 2)
            equation = f"y = {round(x1,4)}$x^2$ + {round(m,4)}x + {round(b,4)}"
            ax.plot(x, x1*x**2 + m*x + b, '-r', label=equation)
        elif self.best_fit==3:
            x = np.linspace(x_values[0],x_values[-1],100)
            x2, x1, m, b = np.polyfit(x_values, scans_mean, 3)
            equation = f"y = {round(x2,4)}$x^3$ + {round(x1,4)}$x^2$ + {round(m,4)}x + {round(b,4)}"
            ax.plot(x, x2*x**3 + x1*x**2 + m*x + b, '-r', label=equation)
        elif self.best_fit==4:
            x = np.linspace(x_values[0],x_values[-1],100)
            x3, x2, x1, m, b = np.polyfit(x_values, scans_mean, 4)
            equation = f"y = {round(x3,4)}$x^4$ + {round(x2,4)}$x^3$ + {round(x1,4)}$x^2$ + {round(m,4)}x + {round(b,4)}"
            ax.plot(x, x3*x**4 + x2*x**3 + x1*x**2 + m*x + b, '-r', label=equation)
        elif self.best_fit==5:
            x = np.linspace(x_values[0],x_values[-1],100)
            x4, x3, x2, x1, m, b = np.polyfit(x_values, scans_mean, 5)
            equation = f"y = {round(x4,4)}$x^5$ + {round(x2,4)}$x^3$ + {round(x1,4)}$x^2$ + {round(m,4)}x + {round(b,4)}"
            ax.plot(x, x4*x**5 + x2*x**3 + x1*x**2 + m*x + b, '-r', label=equation)

        ax.legend()
        plt.show()
        print(scans_mean)

        pass

class Indiv_Plot(object):
    def __init__(self, **kwargs):
        """Instance initialization.

        Args:
        """
        # Logger
        # self.__logger = logger
        # Plotter settings
        for key, value in DEFAULT_CONFIG['indiv_plotter'].items():
            if key in kwargs and kwargs[key]:
                setattr(self, key, kwargs[key])
            else:
                # self.__logger.debug("Using default beacon advertiser "
                #         f"configuration {key}: {value}.")
                setattr(self, key, value)
                print("Default attribute initialized")

        # self.__logger.info("Initialized beacon advertiser.")
        print("Initialized Plotter")

    def plot_indiv(self):
        file_data = pd.read_csv(getattr(self, "file_location"))
        scan_values = file_data["RSSI"].tolist()

        fig1, ax = plt.subplots()
        ax.boxplot(scan_values, vert=False, meanline=True, showmeans=True, meanprops={'linewidth':2.5, 'color':'red'}, medianprops={'linestyle':'None'})

        ax.set_title("Box and Whiskers Plot of RSSI Values")
        ax.set_xlabel('RSSI Values')        
            
        y = [1] * len(scan_values)
        counts = Counter(zip(scan_values,y))
        s = [200*counts[(xx,yy)] for xx,yy in zip(scan_values,y)]
            
        plt.scatter(scan_values, y, s=s, alpha=0.2)
        plt.yticks([])

        plt.show()

        print("Average Value: " + str(statistics.mean(scan_values)))

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
        config['indiv_plotter'] = {**DEFAULT_CONFIG['indiv_plotter'],
                **config['indiv_plotter']}
    # Merge configuration values with command line options
    for key, value in parsed_args.items():
        if value is not None:
            if key in config['all_grapher']:
                config['all_grapher'][key] = value
            if key in config['indiv_plotter']:
                config['indiv_plotter'][key] = value
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
    mode_group.add_argument('-i', '--indiv_plotter', action='store_true',
                            help="Create scatter plot of data in specified file")       
    parser.add_argument('--config_yml', help="Configuration YAML.")
    parser.add_argument('--file_location', help="Path to file")
    parser.add_argument('--scan_prefix', help="Prefix to numbered file (file should be scan_prefix#.csv)")
    parser.add_argument('--graph_title', help="Title of resulting graph")
    parser.add_argument('--x_label', help="Label for y axis")
    parser.add_argument('--y_label', help="Label for x axis")
    parser.add_argument('--best_fit', type=int,
            help="Degree of line of best fit, -1 means no line")
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
        elif parsed_args['indiv_plotter']:
            plotter = Indiv_Plot(**config['indiv_plotter'])
            plotter.plot_indiv()
    except Exception:
        print("Something has gone wrong...oops")
    # finally:
    #     close_logger(logger)

if __name__ == "__main__":
    """Script execution."""
    main(sys.argv[1:])
