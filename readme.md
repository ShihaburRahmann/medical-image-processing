# Medical Imaging Project

This project loads, processes, and visualizes CT scans and segmentation data (such as liver and tumor regions) from DICOM files.

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

This will load the CT volume and segmentation files, verify the data, and display the CT images with the labeled regions using various 2D projection views.
