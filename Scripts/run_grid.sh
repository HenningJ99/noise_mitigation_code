#!/bin/bash

read -p "Working path: " path
read -p "#Galaxies per shear: " galaxy_num
read -p "#Galaxies for Pujol: " pujol_num
read -p "Shear interval: " shear_interval
read -p "Smallest magnitude: " min_mag
read -p "Largest magnitude: " max_mag
read -p "Magnitude bins: " mag_bins
read -p "Time bins: " time_bins
read -p "Reps for improvement error: " reps
read -p "Generation only? (y/n): " gen_only
if [ $gen_only == "n" ]
then
  read -p "Analysis only? (y/n): " analysis_only
fi

# read -p "0 (no shape noise cancel) or 1: " x_tiles
# read -p "0 (no pixel noise cancel) or 1: " y_tiles

# ------------------ MODIFY CONFIG FILE WITH INPUT --------------------------------------------------#
python3 modify_config.py GRID $mag_bins $time_bins $min_mag $max_mag $reps

compare=`echo | awk "{ print ($shear_interval == 0.02)?1 : 0 }"` #Comparison to decide on shear interval

object_num=$((galaxy_num * 20))

if [ $analysis_only == "n" ]
then
  # ------------------ CATALOG GENERATION ------------------------------------------------------------#
  echo "Starting Fit method simulations!"
  python3 grid_simulation.py $object_num $galaxy_num 2 1 $shear_interval $path
  lf_folder=$(ls -td $path/output/grid_simulations/* | head -1)

  echo "Starting Response method simulations!"
  if [[ $compare -eq 1 ]]
  then
    python3 pujol_grid.py $pujol_num 1 CCD_SIM 2 $path
    puj_folder=$(ls -td $path/output/grid_simulations/* | head -1)
  else
    python3 pujol_grid.py $pujol_num 1 CCD_SIM 11 $path
    puj_folder=$(ls -td $path/output/grid_simulations/* | head -1)
  fi
fi

if [ $gen_only == "n" ]
then

  if [ $analysis_only == "y" ]
  then
    read -p "Linear fit folder: " lf_folder
    read -p "Pujol folder: " puj_folder

    lf_folder=$path/output/grid_simulations/$lf_folder
    puj_folder=$path/output/grid_simulations/$puj_folder

    # Check if directories exist
    if [ -d "$lf_folder" ] && [ -d "$puj_folder" ]; then
      ### Take action if $DIR exists ###
      echo "Files found! Continuing"
    else
      ###  Control will jump here if $DIR does NOT exists ###
      echo "Error: ${lf_folder} or ${puj_folder} not found. Can not continue."
      exit 1
    fi
  fi

  # ----------------------- ANALYSIS ------------------------------------------------------------------------------#
  for i in $(seq 0 $((reps-1)))
  do
    python3 grid_analysis.py $object_num $galaxy_num 2 1 $shear_interval $lf_folder $i

    python3 plot_data.py $lf_folder/results_${object_num}_${galaxy_num}_1_1.dat 1 $path
    python3 plot_data.py $lf_folder/results_${object_num}_${galaxy_num}_1_2.dat 1 $path
    python3 plot_data.py $lf_folder/results_${object_num}_${galaxy_num}_2_2.dat 1 $path
  done

  for i in $(seq 0 $((reps-1)))
  do
    if [[ $compare -eq 1 ]]
    then
      python3 pujol_grid_analysis.py $pujol_num 1 2 $puj_folder $path $i
    else
      python3 pujol_grid_analysis.py $pujol_num 1 11 $puj_folder $path $i
    fi
  done

  # --------------------- EXTRACTION FROM OUTPUT FILES -----------------------------------------------------------#
  tail ${path}/output/grid_simulations/fits.txt -n $((4 * reps * time_bins * (mag_bins+1))) >> \
  $path/output/grid_simulations/tmp.txt
  #tail $puj_folder/puyol_results.txt -n $((time_bins * (mag_bins+1))) >> $path/output/grid_simulations/tmp.txt

  # --------------------- UNCERTAINTY BEHAVIOUR PLOTS --------------------------------------------------------------#
  python3 error_plot_grid.py $path M $lf_folder $shear_interval

  tail ${lf_folder}/error_scaling_M.txt -n $((4 * (mag_bins+1))) >> $path/output/plots/binned_improvement_m.txt

  python3 plot_binned_data.py $path/output/plots/binned_improvement_m.txt config_grid.ini $shear_interval GRID

  python3 error_plot_grid.py $path C $lf_folder $shear_interval

  tail ${lf_folder}/error_scaling_C.txt -n $((4 * (mag_bins+1))) >> $path/output/plots/binned_improvement_c.txt

  python3 plot_binned_data.py $path/output/plots/binned_improvement_c.txt config_grid.ini $shear_interval GRID

  # ---------------------- DIRECT OUTPUT FOR BETTER BIAS COMPARISON -------------------------------------------------#
  head $path/output/grid_simulations/tmp.txt -n $((mag_bins+1)) >> $path/output/plots/grid_bias.txt
  head $path/output/grid_simulations/tmp.txt -n $(((mag_bins+1) * time_bins + (mag_bins+1))) | tail -n $((mag_bins+1)) \
  >> $path/output/plots/grid_bias.txt

  head $path/output/grid_simulations/tmp.txt -n $((2*(mag_bins+1) * time_bins + (mag_bins+1))) \
  | tail -n $((mag_bins+1)) >> $path/output/plots/grid_bias.txt

  tail $path/output/grid_simulations/tmp.txt -n $((time_bins * (mag_bins+1))) \
  | head -n $((mag_bins+1)) >> $path/output/plots/grid_bias.txt

  # ------------------------------- PLOT THE BINNED COMPARISON -----------------------------------------------#
  python3 bias_comparison.py $path/output/plots/grid_bias.txt GR

  # ---------------------- REMOVE TEMPORARY FILES -------------------------------------------------------------------#
  rm $path/output/grid_simulations/tmp.txt
  rm $path/output/plots/binned_improvement_m.txt
  rm $path/output/plots/binned_improvement_c.txt
  rm $path/output/plots/grid_bias.txt

fi

echo "Done!"