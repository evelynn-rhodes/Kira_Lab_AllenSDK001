# Kira_Lab_AllenSDK001
Exploraing data of MindScope from the Allen institute by using the AllenSDK package.

Hello!

Welcome to the individual project that I completed in Fall 2025 with the help of Dr. Kira.

The goal of this project was to analyze the Allen Institute’s MindScope data, which can be found here: [Allen Institute Visual Coding: Static Gratings](https://observatory.brain-map.org/visualcoding/stimulus/static_gratings?utm_source=chatgpt.com)

We decided to use this dataset because the Allen Institute used *in vivo* calcium imaging to measure responses in the visual cortex to static and drifting gratings with different orientations, spatial frequencies/temporal frequencies, and phases. Since these stimuli are very similar to those used in the Kira Lab, and because we will also be analyzing *in vivo* calcium imaging data, this dataset provides valuable insight into what we may observe in our own experiments in the future. By analyzing these data, we were able to look for clear patterns in visual cortex activity as the stimulus changed and later compare those patterns to our own recordings. Throughout this project, we also explored potential sources of noise in calcium imaging data, discussed which analytical approaches work best for this type of dataset, and created a foundation for future analysis pipelines that can be adapted for our own lab data.

This project is divided into two main parts: simulated data from Dr. Kira and analysis of the Allen Institute data completed by me (Evelynn Rhodes) and another lab member, Keelyn McAnarney.

For the analysis, the first step was downloading the experimental data. To do this, I used the [AllenSDK](https://allensdk.readthedocs.io/en/latest/?utm_source=chatgpt.com), a Python package designed for accessing and processing Allen Brain Atlas datasets. I found that using Google Colab made the workflow much easier, since it allowed the code to run consistently across different computers without requiring complicated local setup. In this portion of the code, we used the AllenSDK to examine experiment metadata and then selected four experiments from either the static or drifting gratings datasets to download. These downloads included:

* metadata describing the experiment (such as imaging depth),
* the stimulus table, which identifies when and which stimulus was presented,
* the df/f trace table, which contains fluorescence activity traces for each neuron over time, and
* the timestamp table, which aligns the neural activity traces with the stimulus presentations.

All of these files were automatically organized into folders labeled with their corresponding experiment IDs.

After downloading the data, we moved into the analysis stage. This portion of the project was completed in MATLAB because of its strong graphing and visualization capabilities, as well as the fact that it is the primary programming environment used in the Kira Lab.

We first wanted to understand the simplest level of the data before scaling up to population-level analyses. To do this, we began by plotting the fluorescence trace of a single neuron during presentations of a specific stimulus condition. Since each stimulus combination was repeated multiple times, this produced several traces for the same condition. We then averaged these traces together to better visualize the neuron’s typical response to that stimulus.

To examine tuning for an individual neuron, we created heatmaps showing the average peak response for every combination of stimulus variables. In these heatmaps, brighter regions indicated the stimuli to which the neuron was most strongly “tuned.” We compared these tuning plots to the Allen Brain Observatory visualizations to verify whether the same neurons appeared tuned to the same orientations and spatial frequencies.

Next, we expanded the analysis to population responses by constructing a response tensor matrix and visualizing it as a 3D cube. Since it was difficult to identify patterns directly from the 3D representation, we also generated population heatmaps to better examine how neural responses changed with orientation and spatial frequency across many cells.

We then performed Principal Component Analysis (PCA) on the dataset. PCA is best explained through resources such as StatQuest and other educational videos, but in simple terms, it reduces high-dimensional data into a lower-dimensional representation that makes patterns easier to visualize. Instead of viewing the data as a large 3D cube or heatmap, PCA allowed us to examine the responses on a 2D graph. On these plots, we looked for structure in the neural responses as orientation or spatial frequency changed. In particular, when varying orientation, we expected to see looping trajectories in PCA space. While these loops were not as clear or smooth as expected, we were interested in observing how those trajectories shifted as spatial frequency changed.

Because the PCA analysis did not produce patterns as cleanly as we had hoped, we also performed a population distribution analysis. This approach allowed us to examine how many neurons were tuned to each specific stimulus condition. To visualize this bias, we created both histograms and heatmaps. In the histograms, we would expect a relatively even distribution of neurons across bins if tuning preferences were balanced. Similarly, in the heatmaps, we would expect evenly sized bright regions forming a diagonal pattern if responses were uniformly distributed across stimulus conditions.

