"""Web front-end for the Jumpstart pipeline (jumpstart/).

This package does not change anything inside jumpstart/ -- it only
wraps it: a local FastAPI server + browser UI that calls the exact
same functions jumpstart/workflow_demo.py's CLI calls (build_profiles,
register_videos, extract_reference_frame, identify_athletes_in_video,
build_jumps_for_athlete, calculate_jump_parameters, export_results,
...), so anything reviewed/tested on the CLI path behaves identically
here. See README.md in this folder for how to run it.
"""
