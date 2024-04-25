import matplotlib.pyplot as plt
import numpy as np
import modifications as mod
import ipywidgets as ipy
import math

def plot_curve(settings, object_index, modification_type = "CUTOFF", cutoff = 0, ref_time_offset = 0,
               used_bands = ["ALL"], shown_bands = ["ALL"], show_scatter = "with error", isolate_variability = True):
    """
    Predicts and plots the full light curve for a single object, with or without a modification to the observations.

    :param dict settings: A dictionary containing various settings needed for making modifications.
    :param int object_index: Specifies which object by its index in the dataset.
    :param str modification_type: The type of modification that should be made. Options are: "CUTOFF" - Sets a maximum number of observations from a light curve that can be used. "REFERENCE TIME" - Offsets the reference time for the light curve.
    :param int cutoff: The maximum number of points allowed in the light curve. If <=0, there is no cutoff.
    :param float ref_time_offset: An offset to the reference time. At 0, the reference time will be left as is.
    :param list[str] used_bands: A list of the bandpasses that should be included when making predictions.
    :param list[str] shown_bands: A list of the bandpasses that should be visible in the plot.
    :param str show_scatter: Whether or not the observation data should be visible on the plot. Accepted values: "with error", "on", "off"
    :param bool isolate_variability: Restricts the x and y axis to areas containing variability.
    """

    fig, ax = plt.subplots(1, 1, figsize = (12, 5))

    if "ALL" in used_bands: used_bands = settings['all_bands']
    if "ALL" in shown_bands: shown_bands = settings['all_bands']

    if modification_type == "CUTOFF":
        modification = [cutoff] if cutoff > 0 else []
        mod_index = 1 if cutoff > 0 else 0

    elif modification_type == "REFERENCE TIME":
        modification = [ref_time_offset] if ref_time_offset > 0 or ref_time_offset < 0 else []
        mod_index = 1 if ref_time_offset > 0 or ref_time_offset < 0 else 0

    # Modifies the dataset and predicts its light curve
    data = mod.classify_modifications(settings, 1, used_bands, modification_type, modification, object_index)
    object_id = list(data.keys())[3] # TODO: Make this simpler

    # Predictions for the unmodified dataset
    true_class = data[object_id][0]['truth']
    primary_light_curve = data[object_id][0]["light_curve"]
    primary_predicted_curve = data[object_id][0]["predicted_curve"]

    start_time = np.min(primary_predicted_curve.time)
    end_time = max(primary_predicted_curve.time.flatten())

    # Reference time
    primary_ref_time = data[object_id][0]['predictions']['reference_time'] - start_time
    modified_ref_time = data[object_id][mod_index]['predictions']['reference_time'] - start_time
    ref_error = data[object_id][0]['predictions']['reference_time_error']

    # Isolates the variable part of the light curve
    variability_start = start_time#data[object_id][0]['predictions']['reference_time'] - 35
    variability_end = end_time#data[object_id][0]['predictions']['reference_time'] + 80
    variability_range = np.where((data[object_id][0]["light_curve"]["time"] > variability_start)
                            & (data[object_id][0]["light_curve"]["time"] < variability_end))[0]
    start_index = variability_range[0]
    end_index = variability_range[-1]
    
    # Trims the vertical portion of the light curve to only contain visible data
    min_y = min(primary_light_curve["flux"][start_index:end_index]) if show_scatter != "off" else min(primary_predicted_curve.flux.flatten()[start_index:end_index])
    max_y = max(max(primary_predicted_curve.flux.flatten()[start_index:end_index]), max(primary_light_curve["flux"][start_index:end_index]))

    # Only plots bandpasses which were both used in modifications and were set to be shown
    for band in set(shown_bands).intersection(used_bands):
        band_index = settings['all_bands'].index(band)

        # Plots the predicted light curve for the unmodified sample in the given band
        ax.plot((primary_predicted_curve.time - np.min(start_time)), primary_predicted_curve.flux[0][band_index],
                label = f"{len(primary_light_curve)} points (original)",
                linestyle = "--", linewidth = 1, color = settings['band_colors'][band_index])

        light_curve = data[object_id][mod_index]["light_curve"]
        light_curve_mask = light_curve["band"] == band

        # Plot the individual observations and their flux error in the given band
        if show_scatter != "off":
            error = light_curve[light_curve_mask]['fluxerr'] if show_scatter == "with error" else np.zeros(len(light_curve[light_curve_mask]['fluxerr']))
            ax.errorbar(light_curve[light_curve_mask]['time'] - start_time, light_curve[light_curve_mask]['flux'],
                        yerr = error, fmt = '.', label = band, color = settings['band_colors'][band_index])

        # Plot a vertical bar indicating the reference time and it's error for the modified dataset
        ax.axvspan(modified_ref_time - ref_error, modified_ref_time + ref_error, alpha=0.2, color='red')

        # Ensure the unmodified light curves are always plotted
        if mod_index > 0:
            predicted_curve = data[object_id][mod_index]["predicted_curve"]
            ax.plot(predicted_curve.time - start_time, predicted_curve.flux[0][band_index],
                    label = f"{len(light_curve)} points", linewidth = 1.5,
                    color = settings['band_colors'][band_index])
            max_y = max(max(predicted_curve.flux.flatten()[start_index:end_index]), max_y)

    # Restrict the viewed area to the variable portion of the light curve
    if isolate_variability:
        ax.set_xlim(left = variability_start - start_time, right = min(variability_end, end_time) - start_time)
    else:
        ax.set_xlim(left = 0, right = end_time - start_time)
    #print(min_y, max_y)

    ax.set_ylim(top = max_y + (0.1 * max_y), bottom = min_y - (0.1 * abs(max_y)))

    # Plots a dashed bar showing the reference time for the unmodified dataset
    ax.axvline(primary_ref_time, min_y - 2, max_y + 2, color='black', ls="--")

    # What is the predicted class and it's associated probability?
    predicted_class = data[object_id][mod_index]['prediction']
    prediction_probability = data[object_id][mod_index][predicted_class]
    title = f"{(prediction_probability * 100):.2f}% {predicted_class}"

    # If the predicted class is incorrect, what is the predicted probability of the true class?
    if predicted_class != true_class:
        truth_probability = data[object_id][mod_index][true_class]
        title = f"{title} ({(truth_probability * 100):.2f}% {true_class})"

    # Make predicted and true class probabilities the title of each plot
    ax.set_title(title)
    fig.suptitle(object_id)

    plt.show();

def plot_interactable_curve(settings, mask = None):
    """
    Predicts and plots a full light curve for a single object, with interactive options determining how the observation data is modified and visualized.

    :param dict settings: A dictionary containing various settings needed for making modifications.
    """

    data_length = len(settings['dataset'])# if mask is None else len(settings['dataset'][mask])
    #data_indices = range(0, data_length) if mask is None else mask.nonzero()

    # Create widgets for adjusting each of the possible options
    band_list = ["ALL"] + settings['all_bands']
    modification_type = ipy.Dropdown(options = ["CUTOFF", "REFERENCE TIME"], description = "Modification Type:", style={'description_width': 'initial'})
    object_index = ipy.BoundedIntText(min = 0, max = data_length - 1, step = 1, value = 0, description = "Index:",
                                      continuous_update = False, style={'description_width': 'initial'})
    cutoff = ipy.SelectionSlider(options = [0, 64, 48, 32, 24, 16, 12, 8, 6, 4], description = "Cutoff:", continuous_update = False)
    ref_time_offset = ipy.FloatSlider(min = -10, max = 10, step = 0.25, description = "Reference Time Offset:",
                                      continuous_update = False, style={'description_width': 'initial'})
    used_bands = ipy.SelectMultiple(options = band_list, rows = len(band_list), value = ["ALL"], description = "Use Bands:")
    shown_bands = ipy.SelectMultiple(options = band_list, rows = len(band_list), value = ["ALL"], description = "See Bands:")
    show_scatter = ipy.RadioButtons(options = ['with error', 'no error', 'off'], description = "Scatter:")

    # BUG: Checkboxes don't seem to be visible for unknown reasons
    #isolate_variability = widgets.Checkbox(value=True, description="Isolate Variability")

    # Organize the option widgets
    ui = ipy.HBox([
        ipy.VBox([modification_type, object_index, show_scatter]),#, isolate_variability]),
        ipy.VBox([cutoff, used_bands]),
        ipy.VBox([ref_time_offset, shown_bands])
    ])

    # Run the options through the plotter
    output = ipy.interactive_output(
        plot_curve, {
            "settings": ipy.fixed(settings),
            #"object_index": data_indices[object_index],
            "object_index": object_index,
            "modification_type": modification_type,
            "cutoff": cutoff,
            "ref_time_offset": ref_time_offset,
            "used_bands": used_bands,
            "shown_bands": shown_bands,
            "show_scatter": show_scatter,
            #"isolate_variability": isolate_variability
        });

    display(output, ui)

def plot_class_grid(data, xaxis_label = "", classes = "ALL", max_objects = -1):
    """
    Creates grid plots for the given classes, depicting the average probability of a particular class being predicted for it.
    Each column is a modification being made to the dataset, and each row is a possible predicted class. The value in each cell
    is the average of all probabilities given by ParSNIP for the objects to be of that class.

    :param dict data: The classifications and modification data for any number of objects, with each object's ID being the key to the dictionary.
    :param str xaxis_label: The label shown below the x-axis on the plot.
    :param str classes: A comma-separated list of classes contained within a string. For example, "SNIa, SNII" will create two plots, one for each of the given types. A value of "ALL" will create a plot for every possible class type.
    :param int max_objects: The maximum number of objects that each given class type is allowed to consider. A value of -1 results in no maximum.
    """

    # NOTE: Plasticc dataset has no KN objects, but ParSNIP can still predict them. So it must be an included row.
    predictable_classes = { "SNIa" : 0., "SNII": 0, "SLSN-I": 0, "SNIa-91bg": 0, "SNIax": 0, "SNIbc": 0, "TDE": 0., "KN": 0. }

    # Creates a list of the classes that we want to be analyzed
    if classes == "ALL": classes = ["SNIa", "SNII", "SLSN-I", "SNIa-91bg", "SNIax", "SNIbc", "TDE"]
    elif "," in classes: classes = classes.replace(" ", "").split(",")
    else: classes = [classes]

    data_tables = []
    for obj_class in classes:

        data_table = { }
        object_count = 0

        for object_id in data:
            if object_id == "used_bands" or object_id == "used_cutoffs" or object_id == "used_offsets": continue
            if not(data[object_id][0]['truth'] == obj_class): continue
            for index in range(0, len(data[object_id])):
                for key in data[object_id][index]:
                    if not(key in ["modification", "light_curve", "truth", "prediction", "predictions", "predicted_curve"]):
                        modification = str(data[object_id][index]["modification"])
                        if modification == 'None': modification = "99999" # Ensures proper sorting for cutoffs
                        if not(modification in data_table):
                            data_table[modification] = predictable_classes.copy()
                        data_table[modification][key] += data[object_id][index][key]

            # If there is a maximum number of objects, stop there
            object_count += 1
            if 0 < max_objects <= object_count: break

        # Divides each value by the total number of objects included so that its a percentage
        for modification in data_table:
            for key in data_table[modification]:
                data_table[modification][key] /= object_count

        data_tables.append({ "class": obj_class, "data": data_table, "count": object_count})

    # If no objects were found at all, then stop here
    if len(data_tables) == 0:
        print("No data available.")
        return

    # Create a grid, with each plot cooresponding to a specific class type
    cols = 2
    rows = math.ceil(len(data_tables) / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

    # Set up the axis labels for all plots
    column_keys = np.array(list(data_tables[0]["data"].keys()), dtype=np.dtype('U10'))
    sorting = np.argsort(column_keys.astype(float))
    column_labels = column_keys[sorting]
    row_labels = list(data_tables[0]["data"][column_keys[0]])
    column_labels[np.isin(column_labels, ["0", "99999"])] = 'NONE'

    # For each class, create an table of the classifications and modifications
    for i in range(len(data_tables)):
        
        ax.flat[i].set_title(f"{data_tables[i]['class']} ({data_tables[i]['count']} objects)")
        table = np.zeros((len(row_labels), len(column_labels)))

        results = list(data_tables[i]["data"].items())

        for j, (key, values) in enumerate(results):
            for k, value in enumerate(values.values()):
                col = np.where(sorting == j)[0][0]
                ax.flat[i].text(col, k, f"{value:.2f}", ha="center", va="center")
                table[k, col] = value

        # Use class names for y-axis labels and modification values for x-axis labels
        if data_tables[i]["count"] > 0:
            ax.flat[i].set_xticks(np.arange(len(column_labels)), column_labels)
            ax.flat[i].set_yticks(np.arange(len(row_labels)), row_labels)
            ax.flat[i].set_xlabel(xaxis_label)
            ax.flat[i].set_ylabel("Predicted Classification")

        # If no data is available for this class type, indicate so and clear the axes
        else:
            ax.flat[i].text((len(column_labels)-1) / 2, (len(row_labels)-1) / 2, "NO DATA AVAILABLE\nIN SUPPLIED DATASET", ha="center", va="center")
            ax.flat[i].set_axis_off()

        # Display the table as an image
        ax.flat[i].imshow(table, cmap = 'Greens', vmin=0, vmax=1)

    # Delete any blank axes left over by the subplots method
    if cols * rows > len(data_tables): fig.delaxes(ax.flat[-1])

    plt.show()
