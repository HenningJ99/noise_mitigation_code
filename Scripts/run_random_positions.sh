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
read -p "Reps for improvement error: " reps
read -p "Analyse every pujol: " analyse_every_pujol
read -p "Skip first rm points: " skip_first_rm
read -p "Skip first lf points: " skip_first_lf
read -p "Generation only? (y/n): " gen_only
if [ $gen_only == "n" ]
then
  read -p "Analysis only? (y/n): " analysis_only
fi

# read -p "0 (no shape noise cancel) or 1: " x_tiles
# read -p "0 (no pixel noise cancel) or 1: " y_tiles

# -------------------------- MODIFY CONFIG FILE WITH MAG_BINS
python3 modify_config.py RP $mag_bins $min_mag $max_mag $analyse_every_pujol $skip_first_lf $skip_first_rm $reps

compare=`echo | awk "{ print ($shear_interval == 0.02)?1 : 0 }"` #Comparison to decide on shear interval

if [ $analysis_only == "n" ]
then
  echo "Starting global cancellation simulations!"
  # -------------------------- CATALOG GENERATION --------------------------------#
  python3 rp_simulation.py $sim_size $galaxy_num $run_lf $path $shear_interval True
  lf_folder_global=$(ls -td $path/output/rp_simulations/*/ | head -1)

  echo "Starting local cancellation simulations!"
  python3 rp_simulation.py $sim_size $galaxy_num $run_lf $path $shear_interval False
  lf_folder_local=$(ls -td $path/output/rp_simulations/*/ | head -1)

  echo "Starting response method simulations!"
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
    read -p "Linear fit folder (global): " lf_folder_global
    read -p "Linear fit folder (local): " lf_folder_local
    read -p "Pujol folder: " puj_folder

    lf_folder_global=$path/output/rp_simulations/$lf_folder_global
    lf_folder_local=$path/output/rp_simulations/$lf_folder_local
    puj_folder=$path/output/rp_simulations/$puj_folder

    # Check if directories exist
    if [ -d "$lf_folder_global" ] && [ -d "$puj_folder" ] && [ -d "$lf_folder_local" ]; then
      ### Take action if $DIR exists ###
      echo "Files found! Continuing"
    else
      ###  Control will jump here if $DIR does NOT exists ###
      echo "Error: ${lf_folder_global} or ${puj_folder} or ${lf_folder_local} not found. Can not continue."
      exit 1
    fi
  fi

  echo "Analysing the Kron radius blending for $puj_folder"
  if [[ $compare -eq 1 ]]
    then
      python3 kron_radius_blending.py $puj_folder $run_rm 2 RM
    else
      python3 kron_radius_blending.py $puj_folder $run_rm 11 RM
  fi

  for shape_options in $lf_folder_local $lf_folder_global
  do
    echo "Analysing the Kron radius blending for $shape_options"
    python3 kron_radius_blending.py $shape_options $run_lf 20 LF

    for binning in MAG_AUTO GEMS # Do the analysis for GEMS and MAG_AUTO Binning for comparison
    do
      echo "Starting analysis for $shape_options binned in $binning!"
      # -------------------------- INITIAL ANALYSIS -------------------------------------#

      echo "Fit method analysis ..."
      python3 rp_analysis.py $sim_size $galaxy_num $run_lf $path $shear_interval $shape_options $binning

      echo "Plotting and bootstrapping ..."
      python3 catalog_plot.py $run_lf $galaxy_num $path $sim_size $shape_options $binning

      # Avoid overwriting output
      mv $shape_options/fits_analysis.dat $shape_options/fits_analysis_${binning}.dat
      mv $shape_options/analysis.dat $shape_options/analysis_${binning}.dat

      echo "Response method analysis ..."
      if [[ $shape_options == $lf_folder_local ]]
      then
        if [[ $compare -eq 1 ]]
        then
          python3 pujol_rp_analysis.py $sim_size $galaxy_num $run_rm 2 $path $puj_folder $binning
        else
          python3 pujol_rp_analysis.py $sim_size $galaxy_num $run_rm 11 $path $puj_folder $binning
        fi
        mv $puj_folder/analysis.dat $puj_folder/analysis_${binning}.dat
      fi

      # -------------------------- EXTRACT BIASES AND UNCERTAINTIES ---------------------#

      if [ $binning == "MAG_AUTO" ]
      then
        tail $shape_options/fits_analysis_${binning}.dat -n $(((mag_bins+1)*3)) >> $path/output/plots/bias_comparison_rp.txt
      fi


      if [ $binning == "MAG_AUTO" ] && [ $shape_options == $lf_folder_global ]
      then
        tail $puj_folder/analysis_${binning}.dat -n $((mag_bins+1)) >> $path/output/plots/bias_comparison_rp.txt
      fi

      # ------------------------ PRODUCE THE UNCERTAINTY EVOLUTION PLOTS -------------------------#
      python3 error_plot.py $run_lf $run_rm $path M $shear_interval $shape_options $puj_folder $binning
      tail ${shape_options}/error_scaling_${binning}_M.dat -n $((4*(mag_bins+1))) >> $path/output/plots/binned_improvement_rp_m.txt

      python3 error_plot.py $run_lf $run_rm $path C $shear_interval $shape_options $puj_folder $binning
      tail ${shape_options}/error_scaling_${binning}_C.dat -n $((4*(mag_bins+1))) >> $path/output/plots/binned_improvement_rp_c.txt

    done
  done

  echo "Creating the binned improvements and bias comparisons!"
  # ------------------------------ PLOT THE BINNED IMPROVEMENTS ----------------------------------------------#
  python3 plot_binned_data.py $path/output/plots/binned_improvement_rp_m.txt config_rp.ini $shear_interval RP
  python3 plot_binned_data.py $path/output/plots/binned_improvement_rp_c.txt config_rp.ini $shear_interval RP

  # ------------------------------- PLOT THE BINNED COMPARISON -----------------------------------------------#
  python3 bias_comparison.py $path/output/plots/bias_comparison_rp.txt RP

  # ----------------------------- REMOVE THE TEMPORARY FILES ---------------------------------------- #
  rm $path/output/plots/binned_improvement_rp_m.txt
  rm $path/output/plots/binned_improvement_rp_c.txt
  rm $path/output/plots/bias_comparison_rp.txt

fi

echo "Done!"



