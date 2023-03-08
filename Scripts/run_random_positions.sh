#!/bin/bash

read -p "Working path: " path
read -p "Simulation size: " sim_size
read -p "#Galaxies per scene: " galaxy_num
read -p "#Shear interval: " shear_interval
read -p "#Runs LF: " run_lf
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

compare=`echo | awk "{ print ($shear_interval == 0.02)?1 : 0 }"` #Comparison to decide on shear interval

if [ $analysis_only == "n" ]
then
  # -------------------------- CATALOG GENERATION --------------------------------#
  python3 rp_simulation.py $sim_size $galaxy_num $run_lf $path $shear_interval True
  lf_folder_dyn=$(ls -td $path/output/rp_simulations/*/ | head -1)

  python3 rp_simulation.py $sim_size $galaxy_num $run_lf $path $shear_interval False
  lf_folder_stat=$(ls -td $path/output/rp_simulations/*/ | head -1)

  if [[ $compare -eq 1 ]]
  then
    python3 pujol_rp.py $sim_size $galaxy_num $run_rm 2 $path
  else
    python3 pujol_rp.py $sim_size $galaxy_num $run_rm 11 $path
  fi

  puj_folder=$(ls -td $path/output/rp_simulations/*/ | head -1)
fi

if [ $gen_only == "n" ]
then

  if [ $analysis_only == "y" ]
  then
    read -p "Linear fit folder (dynamic): " lf_folder_dyn
    read -p "Linear fit folder (static): " lf_folder_stat
    read -p "Pujol folder: " puj_folder

    lf_folder_dyn=$path/output/rp_simulations/$lf_folder_dyn
    lf_folder_stat=$path/output/rp_simulations/$lf_folder_stat
    puj_folder=$path/output/rp_simulations/$puj_folder

    # Check if directories exist
    if [ -d "$lf_folder_dyn" ] && [ -d "$puj_folder" ] && [ -d "$lf_folder_stat" ]; then
      ### Take action if $DIR exists ###
      echo "Files found! Continuing"
    else
      ###  Control will jump here if $DIR does NOT exists ###
      echo "Error: ${lf_folder_dyn} or ${puj_folder} or ${lf_folder_stat} not found. Can not continue."
      exit 1
    fi
  fi

  # -------------------------- PLOTTING THE SCENES --------------------------------#
  #python3 display_fits.py lf $lf_folder_dyn 32 $((sim_size-32)) 32 $((sim_size-32))
  #python3 display_fits.py lf $lf_folder_stat 32 $((sim_size-32)) 32 $((sim_size-32))
  #python3 display_fits.py puj $puj_folder 32 $((sim_size-32)) 32 $((sim_size-32))
  #counter=4
  for shape_options in $lf_folder_stat $lf_folder_dyn
  do
    for binning in MAG_AUTO GEMS # Do the analysis for GEMS and MAG_AUTO Binning for comparison
    do
      # -------------------------- INITIAL ANALYSIS -------------------------------------#
      python3 rp_analysis.py $sim_size $galaxy_num $run_lf $path $shear_interval $shape_options $binning

      python3 catalog_plot.py $run_lf $galaxy_num $path $sim_size $shape_options $binning

      extraction=$(((mag_bins+1) * (3 * run_lf + run_rm / analyse_every_pujol)))
      if [[ $shape_options == $lf_folder_stat ]]
      then
        if [[ $compare -eq 1 ]]
        then
          python3 pujol_rp_analysis.py $sim_size $galaxy_num $run_rm 2 $path $puj_folder $binning
        else
          python3 pujol_rp_analysis.py $sim_size $galaxy_num $run_rm 11 $path $puj_folder $binning
        fi
      else
        tail ${path}/output/rp_simulations/fits.txt -n $((2 * extraction)) | head -n $(((mag_bins+1) * run_rm / analyse_every_pujol)) >> ${path}/output/rp_simulations/fits.txt
      fi

      # -------------------------- EXTRACT BIASES AND UNCERTAINTIES ---------------------#

      echo $extraction
      tail ${path}/output/rp_simulations/fits.txt -n $((extraction)) >> $path/output/plots/error_with_real/tmp_rm.txt

      head $path/output/plots/error_with_real/tmp_rm.txt -n $(((mag_bins+1)*3*(run_lf))) | tail -n $(((mag_bins+1)*3)) >> $path/output/plots/error_with_real/tmp_rm_bias.txt

      if [ $binning == "MAG_AUTO" ]
      then
        head $path/output/plots/error_with_real/tmp_rm.txt -n $(((mag_bins+1)*3*(run_lf))) | tail -n $(((mag_bins+1)*3)) >> $path/output/plots/error_with_real/bias_comparison_rp.txt
      fi
      # head $path/output/plots/error_with_real/tmp_rm.txt -n $(((mag_bins+1)*3*(run_lf-1))) | tail -n $((mag_bins+1)*3) | tail -n 3 >> $path/output/plots/error_with_real/tmp_rm_bias.txt
      tail $path/output/plots/error_with_real/tmp_rm.txt -n $((mag_bins+1)) >> $path/output/plots/error_with_real/tmp_rm_bias.txt

      if [ $binning == "MAG_AUTO" ] && [ $shape_options == $lf_folder_dyn ]
      then
        tail $path/output/plots/error_with_real/tmp_rm.txt -n $((mag_bins+1)) >> $path/output/plots/error_with_real/bias_comparison_rp.txt
      fi

      # ------------------------ PRODUCE THE UNCERTAINTY EVOLUTION PLOTS -------------------------#
      python3 error_plot.py $run_lf $run_rm $path M $shear_interval
      tail ${path}/output/error_scaling.txt -n $((4*(mag_bins+1))) >> $path/output/plots/tmp_binned_improvement_rm.txt

      python3 error_plot.py $run_lf $run_rm $path C $shear_interval
      tail ${path}/output/error_scaling.txt -n $((4*(mag_bins+1))) >> $path/output/plots/tmp_binned_improvement_rm_c.txt

      rm $path/output/plots/error_with_real/tmp_rm.txt
      # counter=$((counter-1))
    done
  done

  # ------------------------------ PLOT THE BINNED IMPROVEMENTS ----------------------------------------------#
  python3 plot_binned_data.py $path/output/plots/tmp_binned_improvement_rm.txt config_rp.ini $shear_interval RP
  python3 plot_binned_data.py $path/output/plots/tmp_binned_improvement_rm_c.txt config_rp.ini $shear_interval RP

  # ------------------------------- PLOT THE BINNED COMPARISON -----------------------------------------------#
  python3 bias_comparison.py $path/output/plots/error_with_real/bias_comparison_rp.txt RP

  # ----------------------------- REMOVE THE TEMPORARY FILES ---------------------------------------- #
  rm $path/output/plots/tmp_binned_improvement_rm.txt
  rm $path/output/plots/tmp_binned_improvement_rm_c.txt
  rm $path/output/rp_simulations/fits.txt
  rm $path/output/plots/error_with_real/bias_comparison_rp.txt

fi




