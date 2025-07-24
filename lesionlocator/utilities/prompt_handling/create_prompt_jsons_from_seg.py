import os
import cc3d
import json
import argparse

import numpy as np
import SimpleITK as sitk

def save_json(json_data, output_path):
	if not os.path.exists(os.path.dirname(output_path)):
		os.makedirs(os.path.dirname(output_path))

	with open(output_path, 'w') as f:
		json.dump(json_data, f, indent=4)

def create_point_json(stats):
	point_json = {}
	for idx, p in enumerate(stats["centroids"][1:]):
		# check for none
		if any(np.isnan(p)):
			continue

		point_json[idx+1] = {
			"point": [str(round(coord, 2)) for coord in p]
		}
	return point_json

def create_box_json(stats):
	box_json = {}
	for idx, b in enumerate(stats["bounding_boxes"][1:]):
		# check for none
		if any(np.isnan(stats["centroids"][idx+1])):
			continue

		# Bounding boxes are defined as half-open intervals [start, end)
		# to adhere to python indexing. We therefore add 1 to the end
		z_min, z_max, y_min, y_max, x_min, x_max = b
		b = [z_min, z_max+1, y_min, y_max+1, x_min, x_max+1]


		box_json[idx+1] = {
			"box": [str(round(coord, 2)) for coord in b]
		}
	return box_json

def create_prompt_jsons(label_path: str, output_path: str, label_type: str):
	labels = os.listdir(label_path)

	os.makedirs(output_path, exist_ok=True)

	for lbl in labels:
		im = sitk.ReadImage(os.path.join(label_path, lbl))
		im_array = sitk.GetArrayFromImage(im)

		# Get connected components
		if label_type == "semantic":
			im_array = cc3d.connected_components(im_array.astype(np.uint16))
		stats = cc3d.statistics(im_array.astype(np.uint16), no_slice_conversion=True)

		# Create json
		point_json = create_point_json(stats)
		box_json = create_box_json(stats)

		save_json(point_json, os.path.join(output_path, "points", lbl.replace(".nii.gz", ".json")))
		save_json(box_json, os.path.join(output_path, "boxes", lbl.replace(".nii.gz", ".json")))	


def prompt_jsons():
	parser = argparse.ArgumentParser(description='Create prompt jsons')
	parser.add_argument('-i', type=str, help='Folder containing label images (semantic or instance segmentation)')
	parser.add_argument('-o', type=str, help='Output folder. Will be created if it does not exist')
	parser.add_argument('-label_type', type=str, choices=["semantic", "instance"], help='Type of annotation', default="instance")
	args = parser.parse_args()

	create_prompt_jsons(args.i, args.o, args.label_type)