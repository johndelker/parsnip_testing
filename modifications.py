import numpy as np
from astropy.table import Table
import parsnip
import lcdata
from collections import namedtuple
import warnings

# Hide a few warnings that would otherwise fill the page
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = UserWarning)
warnings.filterwarnings("ignore", message = "'verbose' argument is deprecated and will be removed in a future release of LightGBM.")

# Defining a new variable type that is useful later
Curve = namedtuple('Curve', 'time flux results')



def prepare_classifier(predictions_path, save_path = None):
	"""
	Creates a new classifier. If a path is specified, it writes the classifier to that path.

    :param str predictions_path: The file path to the saved predictions data.
    :param str save_path: Optional path where the classifier should be saved. If not specified, the classifier will not be saved.
    :return: The classifier that was created.
	"""

	new_classifier = parsnip.Classifier()
	training_predictions = Table.read(predictions_path)
	new_classifier.train(training_predictions)
	if not (save_path is None):
		new_classifier.write(save_path)
	return new_classifier


def restrict_bands(dataset, bands):
	"""
	For each sample in the dataset, this method removes all points that are not part of the given bands.

    :param lcdata.dataset.Dataset dataset: A  dataset containing light curve data for one or more objects.
    :param list[str] bands: A list of all bandpasses that should be kept. Any bands not part of this list will be removed from the dataset.
    :return: The modified dataset, with unnecessary bands removed.
	"""

	# Makes a copy of the dataset so that the original is left unmodified
	modified_dataset = lcdata.Dataset(dataset.meta.copy(), dataset.light_curves.copy())

	# Group the light curves by band
	for i in range(0, len(modified_dataset)):
		light_curve = dataset.light_curves[i].group_by('band')
		group_count = len(light_curve.groups.indices) - 1

		# Mask out all points that are part of any band that is being removed for this test
		mask = np.ones(len(light_curve), dtype = bool)
		for n in range(0, group_count):
			if light_curve.groups.keys["band"][n] in bands: continue
			mask[light_curve.groups.indices[n]: light_curve.groups.indices[n + 1]] = False

		# Apply the mask and return to the original sorting method to prevent possible issues
		modified_dataset.light_curves[i] = light_curve[mask]
		modified_dataset.light_curves[i].sort("time")

	return modified_dataset


def restrict_observation_count(dataset, max_observations_per_band):
	"""
	Removes observations past the max number of observations specified.

    :param lcdata.dataset.Dataset dataset: A  dataset containing light curve data for one or more objects.
    :param int max_observations_per_band: The maximum number of observations that should be kept per bandpass. Observations past this count will be removed.
    :return: The modified dataset, with observations removed.
	"""

	# Makes a copy of the dataset so that the original is left unmodified
	modified_dataset = lcdata.Dataset(dataset.meta.copy(), dataset.light_curves.copy())

	# For each object in the dataset, limit the number of observations in each band
	for i in range(0, len(modified_dataset)):
		# Group the light curves by band so that each band can be manipulated individually
		light_curve = dataset.light_curves[i].group_by('band')
		group_count = len(light_curve.groups.indices) - 1

		# Create a mask to remove points from each group/band
		mask = np.ones(len(light_curve), dtype = bool)
		for n in range(0, group_count):
			lower_bound = light_curve.groups.indices[n]
			upper_bound = light_curve.groups.indices[n + 1]

			# Otherwise, only mask out the portion that would be above the point cutoff
			# Note: A cutoff of "None" retains all original observations
			if max_observations_per_band is not None and lower_bound + max_observations_per_band < upper_bound:
				mask[lower_bound + max_observations_per_band: upper_bound] = False

		# Apply the mask and return to the original sorting method to prevent possible issues
		modified_dataset.light_curves[i] = light_curve[mask]
		modified_dataset.light_curves[i].sort("time")

	return modified_dataset

def classify_offsets(settings, sample_size, used_bands, offsets = None, object_index = None):
	"""
	Classifies a dataset or object multiple times using given reference time offsets to study the effects of reference time on the results of ParSNIP.

    :param dict settings: A dictionary containing various settings needed for making modifications.
    :param int sample_size: The number of objects to include from the dataset.
    :param list[str] used_bands: A list of the bandpasses that should be included in the test. Any bands not included will be removed from the dataset.
    :param list[float] offsets: A list of reference time offsets to apply to the dataset. If empty, no modifications will be made.
    :param int object_index: If only one object is being modified, this specifies which object by its index in the dataset.
    :return: A dictionary of every modified object with a list of its modifications and their results.
	"""
	return classify_modifications(settings, sample_size, used_bands, "REFERENCE TIME", offsets, object_index)

def classify_cutoffs(settings, sample_size, used_bands, cutoffs = None, object_index = None):
	"""
	Classifies a dataset or object multiple times using given cutoffs0 to study the effects of limited observation counts on the results of ParSNIP.

    :param dict settings: A dictionary containing various settings needed for making modifications.
    :param int sample_size: The number of objects to include from the dataset.
    :param list[str] used_bands: A list of the bandpasses that should be included in the test. Any bands not included will be removed from the dataset.
    :param list[int] cutoffs: A list of observation cutoffs to apply to the dataset. If empty, no modifications will be made.
    :param int object_index: If only one object is being modified, this specifies which object by its index in the dataset.
    :return: A dictionary of every modified object with a list of its modifications and their results.
	"""
	return classify_modifications(settings, sample_size, used_bands, "CUTOFF", cutoffs, object_index)

def classify_modifications(settings, sample_size, used_bands, modification_type = "CUTOFF", modifications = None, object_index = None):
	"""
	Classifies a dataset or object multiple times using given modifications to study the effects of those modifications on the results of ParSNIP.

    :param dict settings: A dictionary containing various settings needed for making modifications.
    :param int sample_size: The number of objects to include from the dataset.
    :param list[str] used_bands: A list of the bandpasses that should be included in the test. Any bands not included will be removed from the dataset.
    :param str modification_type: The type of modification that should be made. Options are: "CUTOFF" - Sets a maximum number of observations from a light curve that can be used. "REFERENCE TIME" - Offsets the reference time for the light curve.
    :param list[int or float] modifications: A list of modification values to apply to the dataset. If empty, no modifications will be made.
    :param int object_index: If only one object is being modified, this specifies which object by its index in the dataset.
    :return: A dictionary of every modified object with a list of its modifications and their results.
	"""

	# Ensure the modification type is valid
	if not (modification_type in ["REFERENCE TIME", "CUTOFF"]):
		print(f"Modification type '{modification_type}' is not recognized.")
		return None

	# If only looking at one object, grab that part of the dataset; otherwise, grab the first objects up to the sample size
	dataset = settings["dataset"][0:sample_size] if object_index is None else settings["dataset"][object_index]

	# Remove unused bands from the dataset
	dataset = restrict_bands(dataset, used_bands)

	# Ensure the first classification is done on the dataset without any modifications
	if modifications is None: modifications = []
	if modification_type == "CUTOFF": modifications.insert(0, None)
	elif modification_type == "REFERENCE TIME": modifications.insert(0, 0)

	# Creates a dictionary for easily analyzing the results of varying modifications to the dataset
	# TODO: Revamp how classified_data is stored so that we can store any amount of metadata without
	#       needing to adjust indices in various places
	classified_data = {
		"used_bands": used_bands,
		"used_cutoffs": modifications if modification_type == "CUTOFF" else None,
		"used_offsets": modifications if modification_type == "REFERENCE TIME" else None
	}

	# Classify the dataset for each modification
	for modification in modifications:

		# If modifying observation count, remove points from the light curves for each object in the dataset before making predictions
		if modification_type == "CUTOFF":
			modified_dataset = restrict_observation_count(dataset, modification)
			predictions = settings["model"].predict_dataset(modified_dataset)

		# If modifying the reference time, offset the time while making predictions
		elif modification_type == "REFERENCE TIME":
			modified_dataset = lcdata.Dataset(dataset.meta.copy(), dataset.light_curves.copy())
			predictions = settings["model"].predict_dataset(modified_dataset, False, modification)

		# Classify the dataset for this modification based on the resulting predictions
		classifications = settings["classifier"].classify(predictions)

		# For every object in the dataset, store the classification and relevant info
		for index in range(0, len(predictions)):

			# Stores on a per-object basis to easily compare how an individual object is affected by modifications
			obj = classifications["object_id"][index]
			object_info = classified_data.get(obj) or []

			# Find the classification with the highest probability
			top_prediction = None
			for c in classifications.colnames:
				if c == "object_id": continue
				if top_prediction is None or classifications[c][index] > classifications[top_prediction][index]:
					top_prediction = c

			# Predict the full light curve based on modified samples
			if modification_type == "CUTOFF":
				predicted_curve = settings["model"].predict_light_curve(modified_dataset.light_curves[index], sample=False, ref_time_offset=0)
			elif modification_type == "REFERENCE TIME":
				predicted_curve = settings["model"].predict_light_curve(modified_dataset.light_curves[index], sample=False, ref_time_offset=modification)

			# Collects all the most important info from dataset, predictions, and classifications into one place
			object_info.append({
				"modification": modification,
				"truth": modified_dataset.meta['type'][index],
				"prediction": top_prediction,
				"predictions": predictions[index],
				"light_curve": modified_dataset.light_curves[index],
				"predicted_curve": Curve(predicted_curve[0], predicted_curve[1], predicted_curve[2]),
				"SNIa": classifications["SNIa"][index],
				"SNII": classifications["SNII"][index],
				"SLSN-I": classifications["SLSN-I"][index],
				"SNIa-91bg": classifications["SNIa-91bg"][index],
				"SNIax": classifications["SNIax"][index],
				"SNIbc": classifications["SNIbc"][index],
				"TDE": classifications["TDE"][index],
				"KN": classifications["KN"][index]
			})

			classified_data.update({ obj: object_info })

	return classified_data
