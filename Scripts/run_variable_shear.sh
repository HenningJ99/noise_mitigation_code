#!/bin/bash

read -p "Working path: " path
read -p "Simulation size: " sim_size
read -p "#Galaxies per scene: " galaxy_num
read -p "#Runs RM: " run_rm
read -p "Magnitude bins: " mag_bins
read -p "Smallest magnitude: " min_mag
read -p "Largest magnitude: " max_mag
read -p "Analyse every pujol: " analyse_every_pujol
read -p "Generation only? (y/n): " gen_only
if [ $gen_only == "n" ]
then
  read -p "Analysis only? (y/n): " analysis_only
fi

# read -p "0 (no shape noise cancel) or 1: " x_tiles
# read -p "0 (no pixel noise cancel) or 1: " y_tiles

# -------------------------- MODIFY CONFIG FILE WITH MAG_BINS
python3 modify_config.py RP $mag_bins $min_mag $max_mag $analyse_every_pujol

if [ $analysis_only == "n" ]
then
  # -------------------------- CATALOG GENERATION --------------------------------#

  python3 pujol_rp.py $sim_size $galaxy_num $run_rm 2 $path zero
  puj_zero_folder=$(ls -td $path/output/rp_simulations/*/ | head -1)

  python3 pujol_rp.py $sim_size $galaxy_num $run_rm 2 $path rand
  puj_rand_folder=$(ls -td $path/output/rp_simulations/*/ | head -1)

fi

if [ $gen_only == "n" ]
then

  if [ $analysis_only == "y" ]
  then
    read -p "Zero folder: " puj_zero_folder
    read -p "Rand folder: " puj_rand_folder

    puj_zero_folder=$path/output/rp_simulations/$puj_zero_folder
    puj_rand_folder=$path/output/rp_simulations/$puj_rand_folder

    # Check if directories exist
    if [ -d "$puj_rand_folder" ] && [ -d "$puj_zero_folder" ]; then
      ### Take action if $DIR exists ###
      echo "Files found! Continuing"
    else
      ###  Control will jump here if $DIR does NOT exists ###
      echo "Error: ${puj_rand_folder} or ${puj_zero_folder} not found. Can not continue."
      exit 1
    fi
  fi


  # -------------------------- INITIAL ANALYSIS -------------------------------------#
  python3 pujol_rp_analysis.py $sim_size $galaxy_num $run_rm 2 $path $puj_zero_folder MAG_AUTO

  python3 pujol_rp_analysis.py $sim_size $galaxy_num $run_rm 2 $path $puj_rand_folder MAG_AUTO

  # -------------------------- DIFFERENCE ANALYSIS -----------------------------------#
  python3 variable_shear_diff.py $puj_zero_folder/meas_arrays.p $puj_rand_folder/meas_arrays.p $run_rm


fi




