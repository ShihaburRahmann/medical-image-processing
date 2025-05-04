import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


REF_DIR = Path("/Users/shihab/UIB/Medical Image Processing/medical-image-processing/2022/11_AP_Ax5.00mm")
INP_DIR = Path("/Users/shihab/UIB/Medical Image Processing/medical-image-processing/2022/30_EQP_Ax5.00mm")

LIVER_PATH = Path("/Users/shihab/UIB/Medical Image Processing/medical-image-processing/2022/11_AP_Ax5.00mm_ManualROI_Liver.dcm")
TUMOR_PATH = Path("/Users/shihab/UIB/Medical Image Processing/medical-image-processing/2022/11_AP_Ax5.00mm_ManualROI_Tumor.dcm")


# return CT volume (z, y, x) and voxel spacing (dz, dy, dx)
def load_dicom_volume(folder):
    # collect all DICOM files in the folder
    dcms = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.dcm')]

    arrays, z_pos = [], []
    for f in dcms:
        # read slice
        ds = pydicom.dcmread(f)                     
        arrays.append(ds.pixel_array.astype(np.int16))

        # slice z‑position
        z_pos.append(float(ds.ImagePositionPatient[2]))  

    # sort slices by z and stack into a 3‑D volume
    order = np.argsort(z_pos)
    vol = np.stack([arrays[i] for i in order], axis=0)
    vol = vol[:-1]

    # voxel spacing (dz, dy, dx)
    dz = float(ds.SliceThickness)
    dy, dx = map(float, ds.PixelSpacing)
    return vol, (dz, dy, dx)


# check if all DICOM slices belong to a single acquisition
def verify_single_acquisition(dicom_dir):
    dicom_dir = Path(dicom_dir)

    # get all DICOM files in folder
    dcm_files = list(dicom_dir.glob("*.dcm")) 

    series_uids = set()
    acquisition_numbers = set()

    for dcm_file in dcm_files:
        # read DICOM header
        ds = pydicom.dcmread(dcm_file)  
        series_uids.add(ds.SeriesInstanceUID)
        acquisition_numbers.add(ds.AcquisitionNumber)

    print(f"Unique Series Instance UIDs: {len(series_uids)}")
    print(f"Unique Acquisition Numbers: {len(acquisition_numbers)}")

    # check if only one series
    is_single_acquisition = len(series_uids) == 1
    if is_single_acquisition:
        print("SINGLE AQUISITION VERIFICATION: Dataset appears to contain a SINGLE acquisition.")
    else:
        print("SINGLE AQUISITION VERIFICATION: Dataset appears to contain MULTIPLE acquisitions.")
    return is_single_acquisition


# rasterize a DICOM-SEG file onto the CT volume grid (z, y, x)
def build_mask_volume(ct_dir, seg_path):
    CT_DIR = Path(ct_dir)

    # load and sort CT slices by z-position
    ct_files = sorted(
        CT_DIR.glob("*.dcm"),
        key=lambda f: float(pydicom.dcmread(f).ImagePositionPatient[2])
    )
    ct_datasets = [pydicom.dcmread(f) for f in ct_files]
    ct_zs = np.array([float(ds.ImagePositionPatient[2]) for ds in ct_datasets])

    # load segmentation and extract frames
    ds_seg = pydicom.dcmread(seg_path)
    frames = ds_seg.pixel_array

    # get z-positions for each segmentation frame
    seg_zs = np.array([
        float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
        for fg in ds_seg.PerFrameFunctionalGroupsSequence
    ])

    # initialize empty mask volume
    n_slices = len(ct_datasets)
    H, W = frames.shape[1], frames.shape[2]
    mask_volume = np.zeros((n_slices, H, W), dtype=bool)

    # map each SEG frame to the closest CT slice
    for i, z in enumerate(seg_zs):
        idx = int(np.argmin(np.abs(ct_zs - z)))
        mask_volume[idx] = (frames[i] > 0)

    return mask_volume


def visualize(base: List[np.ndarray], spacing, titles: List[str],
              masks: List[Optional[np.ndarray]] = None,
              alpha=0.3, cmaps=("autumn", "cool")):

    # calculate physical extents for each plane
    dz, dy, dx = spacing
    exts = ([0, dx * base[0].shape[1], 0, dy * base[0].shape[0]],
            [0, dx * base[1].shape[1], 0, dz * len(base[1])],
            [0, dy * base[2].shape[1], 0, dz * len(base[2])])

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    for i, ax in enumerate(axs):
        # show CT slice
        ax.imshow(base[i], cmap='bone', extent=exts[i], origin='lower')  
        if masks:
            for m_idx, m in enumerate(masks):
                if m is not None:
                    ax.imshow(m[i], cmap=cmaps[m_idx % len(cmaps)], alpha=alpha,
                              extent=exts[i], origin='lower')
        ax.set_title(titles[i])
        ax.axis('off')
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def show_midplanes(vol, spacing, mask1=None, mask2=None, alpha=0.25):
    # extract center slices in all three planes
    z, y, x = np.array(vol.shape) // 2
    base = [vol[z], vol[:, y, :], vol[:, :, x]]
    masks = None
    if mask1 is not None or mask2 is not None:
        masks = [[m[z], m[:, y, :], m[:, :, x]] if m is not None else None for m in (mask1, mask2)]
    visualize(base, spacing, [f'Axial z={z}', f'Coronal y={y}', f'Sagittal x={x}'], masks, alpha)


def show_mip_planes(vol, spacing, mask1=None, mask2=None, alpha=0.25):
    # compute maximum intensity projections in three planes
    base = [vol.max(0), vol.max(1), vol.max(2)]
    masks = None
    if mask1 is not None or mask2 is not None:
        masks = [[m.max(0), m.max(1), m.max(2)] if m is not None else None for m in (mask1, mask2)]
    visualize(base, spacing, ['Axial MIP', 'Coronal MIP', 'Sagittal MIP'], masks, alpha)


def show_aip_planes(vol, spacing, mask1=None, mask2=None, alpha=0.25):
    # compute average intensity projections in three planes
    base = [vol.mean(0), vol.mean(1), vol.mean(2)]
    masks = None
    if mask1 is not None or mask2 is not None:
        masks = [[m.max(0), m.max(1), m.max(2)] if m is not None else None for m in (mask1, mask2)]
    visualize(base, spacing, ['Axial AIP', 'Coronal AIP', 'Sagittal AIP'], masks, alpha)


# save DICOM header to text file
def save_header(path, out_txt):
    ds = pydicom.dcmread(path)
    with open(out_txt, 'w') as f:
        f.write(str(ds))


if __name__ == "__main__":

    # generate headers for each type of file
    save_header(next(REF_DIR.glob("*.dcm")), "header_ct.txt")
    save_header(TUMOR_PATH, "header_tumor.txt")
    save_header(LIVER_PATH, "header_liver.txt")

    # load the reference directory
    volume, spacing_mm = load_dicom_volume(REF_DIR)

    # verify single aquisition
    is_single = verify_single_acquisition(REF_DIR)

    # generate masks
    mask_volume_liver = build_mask_volume(REF_DIR, LIVER_PATH)
    mask_volume_tumor = build_mask_volume(REF_DIR, TUMOR_PATH)

    # visualize
    show_midplanes(volume, spacing_mm, mask_volume_liver, mask_volume_tumor)
    show_mip_planes(volume, spacing_mm, mask_volume_liver, mask_volume_tumor)
    show_aip_planes(volume, spacing_mm, mask_volume_liver, mask_volume_tumor)