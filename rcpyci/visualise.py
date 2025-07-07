import json
import os
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from rcpyci.im_ops import read_image
from rcpyci.interface import process_condition, process_participant
from rcpyci.pipelines import full_pipeline


def load_config(config_file):
    """Load the configuration from the provided file."""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}

def save_config(config, config_file):
    """Save the updated configuration to the provided file."""
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def update_pipeline_with_config(pipeline, config):
    return [(func, {**default_kwargs, **config.get(str(i), {})}) for i, (func, default_kwargs) in enumerate(pipeline)]

def update_param(step_index, key, config_file):
    """Update a parameter in the config file."""
    config = load_config(config_file)
    new_value = st.session_state[f"{step_index}_{key}"]
    config.setdefault(str(step_index), {})[key] = new_value
    save_config(config, config_file)

def rescale_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val) if max_val > min_val else image

def apply_cluster_overlays(image, pos_cluster, neg_cluster):
    """
    Apply red and blue overlays to an image with full opacity (no transparency).
    """
    overlayed_image = np.stack([image, image, image], axis=-1).astype(np.float32)

    if pos_cluster.max() > 1:
        pos_cluster = pos_cluster / pos_cluster.max()
    if neg_cluster.max() > 1:
        neg_cluster = neg_cluster / neg_cluster.max()

    red_overlay = np.zeros_like(overlayed_image, dtype=np.float32)
    blue_overlay = np.zeros_like(overlayed_image, dtype=np.float32)

    red_overlay[..., 0] = pos_cluster
    blue_overlay[..., 2] = neg_cluster

    overlayed_image += red_overlay + blue_overlay

    return np.clip(overlayed_image, 0, 1)


def streamlit_app(data: pd.DataFrame,
                  base_face_path: str,
                  experiment_path: str,
                  pipelines: List[Tuple[Callable, dict]] = full_pipeline,
                  stimuli_params: np.ndarray = None,
                  patches: np.ndarray = None,
                  patch_idx: np.ndarray = None,
                  n_trials: int = 770,
                  n_scales: int = 5,
                  gabor_sigma: int = 5,
                  noise_type: str = 'sinusoid',
                  seed: int = 1):
    st.set_page_config(layout="wide")
    st.title("Dynamic Condition & Participant Selection")

    base_image = read_image(base_face_path, grayscale=True)

    # Condition Selection (First Row)
    st.subheader("Select Condition")
    condition_alias_map = dict(zip(data["condition_alias"], data["condition"], strict=False))
    selected_condition_alias = st.selectbox("Condition", sorted(data["condition_alias"].unique()))

    # Get the actual condition corresponding to the selected alias
    selected_condition = condition_alias_map[selected_condition_alias]

    # Participant Selection (Second Row)
    st.subheader("Select Participant")
    filtered_participants = sorted(data[data["condition"] == selected_condition]["participant_id"].unique().tolist())
    participant_options = ["All"] + filtered_participants
    selected_participant = st.selectbox("Participant", participant_options)

    st.write(f"**{len(filtered_participants)}** participants available for this condition.")

    # Process Images
    process_func = process_participant if selected_participant != "All" else process_condition
    results = process_func(
        selected_participant if selected_participant != "All" else selected_condition,
        data, base_image, stimuli_params=stimuli_params, n_trials=n_trials,
        experiment_path=experiment_path, pipelines=pipelines
    )

    # Visualization config
    config_file = st.text_input("Enter the config file name:", value=f"{str(os.path.join(experiment_path, 'visualization_config.json'))}")

    config = load_config(config_file)
    pipeline = update_pipeline_with_config(pipelines, config)

    # Process Images
    process_func = process_participant if selected_participant != "All" else process_condition
    results = process_func(
        selected_participant if selected_participant != "All" else selected_condition,
        data, base_image, stimuli_params=stimuli_params, n_trials=n_trials,
        experiment_path=experiment_path, pipelines=pipeline
    )

    ##################################################################
    #                           BASE                                 #
    st.subheader(f"Condition: {selected_condition_alias}")
    
    anti_ci = results[1]['combined']
    ci = results[3]['combined']

    with st.container():
        col_pipeline = st.columns(4)
        for i in range(4):
            with col_pipeline[i]:
                display_pipeline_step(i, pipeline, config, config_file)

        col_images = st.columns(3)
        col_images[0].image(base_image, caption="Base Image", use_container_width=False)
        col_images[1].image(ci, caption="CI Image", use_container_width=True)
        col_images[2].image(anti_ci, caption="Anti-CI Image", use_container_width=True)

    ##################################################################
    #                           PIXELS                               #
    st.subheader(f"Condition: {selected_condition_alias} - Pixel Analysis")

    zscores_ci = results[4]['zscore_image']
    ci_zmap_positive_pixels = results[5]['significant_pixels_positive']
    ci_zmap_negative_pixels = results[5]['significant_pixels_negative']
    ci_pixel_overlay = apply_cluster_overlays(ci, ci_zmap_positive_pixels, ci_zmap_negative_pixels)

    ci_zmap_positive_cluster = results[6]['significant_clusters_positive']
    ci_zmap_negative_cluster = results[6]['significant_clusters_negative']
    ci_cluster_overlay = apply_cluster_overlays(ci, ci_zmap_positive_cluster, ci_zmap_negative_cluster)

    with st.container():
        col_pipeline = st.columns(3)
        for i in range(4, 7):
            with col_pipeline[i - 4]:
                display_pipeline_step(i, pipeline, config, config_file)

        col_images = st.columns(3)
        col_images[0].image(rescale_image(zscores_ci), caption="Z-scores on CI", use_container_width=True)
        col_images[1].image(ci_pixel_overlay, caption="Significant pixels", use_container_width=True)
        col_images[2].image(ci_cluster_overlay, caption="Significant clusters (Blue/Red negative/positive z-scores)", use_container_width=True)

    ##################################################################
    #                           CLUSTERS                             #
    st.subheader(f"Condition: {selected_condition_alias} - Cluster Analysis")

    z_scores_stim_param = results[7]['zscore_image']
    zmap_rcf_pos_cluster = results[8]['significant_clusters_positive']
    zmap_rcf_neg_cluster = results[8]['significant_clusters_negative']

    base_overlay = apply_cluster_overlays(base_image, zmap_rcf_pos_cluster, zmap_rcf_neg_cluster)
    ci_overlay = apply_cluster_overlays(ci, zmap_rcf_pos_cluster, zmap_rcf_neg_cluster)

    with st.container():
        col_pipeline = st.columns(2)
        for i in range(7, 9):
            with col_pipeline[i - 7]:
                display_pipeline_step(i, pipeline, config, config_file)

        col_images = st.columns(3)
        col_images[0].image(rescale_image(z_scores_stim_param), caption="Z-scores on stim params", use_container_width=True)
        col_images[1].image(base_overlay, caption="Base Image significant clusters", use_container_width=True)
        col_images[2].image(ci_overlay, caption="CI significant clusters", use_container_width=True)

        

def display_pipeline_step(i, pipeline, config, config_file):
    """Helper function to display pipeline steps."""
    func, default_kwargs = pipeline[i]
    with st.expander(f"Step {i + 1}: {func.__name__}", expanded=True):
        for key, val in default_kwargs.items():
            current_value = config.get(str(i), {}).get(key, val)
            widget_key = f"{i}_{key}"
            create_widget(key, val, current_value, widget_key, i, config_file)

def create_widget(key, val, current_value, widget_key, step_index, config_file):
    """
    Helper function to create the appropriate widget based on value type.
    """
    if isinstance(val, bool):
        return st.checkbox(key, value=current_value, key=widget_key, on_change=update_param, args=(step_index, key, config_file))
    elif isinstance(val, int):
        return st.number_input(key, value=current_value, step=1, key=widget_key, on_change=update_param, args=(step_index, key, config_file))
    elif isinstance(val, float):
        return st.number_input(key, value=current_value, step=0.05, format="%.5f", key=widget_key, on_change=update_param, args=(step_index, key, config_file))
    elif isinstance(val, str):
        return st.text_input(key, value=current_value, key=widget_key, on_change=update_param, args=(step_index, key, config_file))
    else:
        st.write(f"Unsupported type for {key}")
        return None
