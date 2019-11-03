#! /usr/bin/env bash

#root_data_dir should point to the directory that contains the data to be normalized, e.g., /Users/mn/data/testdata
root_data_dir=$1

python normalize_image_intensities.py --files_to_compute_cdf_from_as_json cdf_datafiles.json --save_average_cdf_to_file average_cdf.pt

# now compute the actual normalizations

python normalize_image_intensities.py --load_average_cdf_from_file average_cdf.pt --directory_to_normalize ${root_data_dir}/CUMC12/brain_affine_icbm --desired_output_directory CUMC12_normalized
mv CUMC12_normalized ${root_data_dir}/CUMC12/brain_affine_icbm_hist_matched

python normalize_image_intensities.py --load_average_cdf_from_file average_cdf.pt --directory_to_normalize ${root_data_dir}/IBSR18/brain_affine_icbm --desired_output_directory IBSR18_normalized 
mv IBSR18_normalized ${root_data_dir}/IBSR18/brain_affine_icbm_hist_matched

python normalize_image_intensities.py --load_average_cdf_from_file average_cdf.pt --directory_to_normalize ${root_data_dir}/LPBA40/brain_affine_icbm --desired_output_directory LPBA40_normalized
mv LPBA40_normalized ${root_data_dir}/LPBA40/brain_affine_icbm_hist_matched

python normalize_image_intensities.py --load_average_cdf_from_file average_cdf.pt --directory_to_normalize ${root_data_dir}/MGH10/brain_affine_icbm --desired_output_directory MGH10_normalized
mv MGH10_normalized ${root_data_dir}/MGH10/brain_affine_icbm_hist_matched

# extract 2D versions

python extractSlicesFrom3DDataSet.py --dataset_directory ${root_data_dir}/IBSR18 --image_subdirectory brain_affine_icbm_hist_matched --output_directory ${root_data_dir}/IBSR18_2d --slice_dim 0 --create_pdfs

python extractSlicesFrom3DDataSet.py --dataset_directory ${root_data_dir}/IBSR18 --image_subdirectory brain_affine_icbm --output_directory ${root_data_dir}/IBSR18_2d --slice_dim 0 --create_pdfs

python extractSlicesFrom3DDataSet.py --dataset_directory ${root_data_dir}/CUMC12 --image_subdirectory brain_affine_icbm_hist_matched --output_directory ${root_data_dir}/CUMC12_2d --slice_dim 0 --create_pdfs

python extractSlicesFrom3DDataSet.py --dataset_directory ${root_data_dir}/CUMC12 --image_subdirectory brain_affine_icbm --output_directory ${root_data_dir}/CUMC12_2d --slice_dim 0 --create_pdfs

python extractSlicesFrom3DDataSet.py --dataset_directory ${root_data_dir}/LPBA40 --image_subdirectory brain_affine_icbm_hist_matched --output_directory ${root_data_dir}/LPBA40_2d --slice_dim 0 --create_pdfs

python extractSlicesFrom3DDataSet.py --dataset_directory ${root_data_dir}/LPBA40 --image_subdirectory brain_affine_icbm --output_directory ${root_data_dir}/LPBA40_2d --slice_dim 0 --create_pdfs

python extractSlicesFrom3DDataSet.py --dataset_directory ${root_data_dir}/MGH10 --image_subdirectory brain_affine_icbm_hist_matched --output_directory ${root_data_dir}/MGH10_2d --slice_dim 0 --create_pdfs

python extractSlicesFrom3DDataSet.py --dataset_directory ${root_data_dir}/MGH10 --image_subdirectory brain_affine_icbm --output_directory ${root_data_dir}/MGH10_2d --slice_dim 0 --create_pdfs







