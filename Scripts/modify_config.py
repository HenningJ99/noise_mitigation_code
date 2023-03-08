import configparser
import sys

if sys.argv[1] == "GRID":
    config = configparser.ConfigParser()
    config.read('config_grid.ini')
    config.set('SIMULATION', 'bins_mag', sys.argv[2])
    config.set('SIMULATION', 'time_bins', sys.argv[3])
    config.set('SIMULATION', 'min_mag', sys.argv[4])
    config.set('SIMULATION', 'max_mag', sys.argv[5])

    with open('config_grid.ini', 'w') as configfile:
        config.write(configfile)

elif sys.argv[1] == "RP":
    config = configparser.ConfigParser()
    config.read('config_rp.ini')
    config.set('SIMULATION', 'bins_mag', sys.argv[2])
    config.set('SIMULATION', 'min_mag', sys.argv[3])
    config.set('SIMULATION', 'max_mag', sys.argv[4])
    config.set('SIMULATION', 'puj_analyse_every', sys.argv[5])

    with open('config_rp.ini', 'w') as configfile:
        config.write(configfile)
