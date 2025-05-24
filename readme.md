# Medical Imaging Project

This project loads, processes, and visualizes CT scans and segmentation data (such as liver and tumor regions) from DICOM files. It creates an animation of the CT volume along with the segmentations. It also performs segmentation using the bounding box of the ground truth tumor mask using two algorithms (Region Growing and Watershed) and compares the results.

## Setup

1. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Before running the project, make sure to update the directory and file paths at the top of `main.py` to match the location of your data on your system.

## Run

To run the project:

```bash
python main.py
```

This will load the CT volume and segmentation files and perform the visualization as well as the segmentation tasks.
