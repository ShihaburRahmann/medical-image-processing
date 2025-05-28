import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import matplotlib.animation as animation
import scipy.ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt, label
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score
from skimage.filters import threshold_otsu

REF_DIR = Path("/Users/shihab/UIB/Medical Image Processing/medical-image-processing/2022/11_AP_Ax5.00mm")
INP_DIR = Path("/Users/shihab/UIB/Medical Image Processing/medical-image-processing/2022/30_EQP_Ax5.00mm")

LIVER_PATH = Path("/Users/shihab/UIB/Medical Image Processing/medical-image-processing/2022/11_AP_Ax5.00mm_ManualROI_Liver.dcm")
TUMOR_PATH = Path("/Users/shihab/UIB/Medical Image Processing/medical-image-processing/2022/11_AP_Ax5.00mm_ManualROI_Tumor.dcm")


# rotate the volume around the axial (Z) plane
def rotate_on_axial_plane(vol: np.ndarray, angle: float) -> np.ndarray:
    return scipy.ndimage.rotate(vol, angle, axes=(1, 2), reshape=False)

# compute maximum intensity projection along the given axis
def max_projection(vol: np.ndarray, axis: int) -> np.ndarray:
    return vol.max(axis=axis)

# generate a rotating MIP animation with mask overlays
def animate_rotating_MIP_planes(vol, spacing, mask1=None, mask2=None):
    alpha = 0.35
    colors = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    cm = plt.get_cmap("gray")

    # rotation angles for 16 projections
    angles = np.linspace(0, 360 * 31 / 32, 32)

    fig, ax = plt.subplots()
    ax.axis("off")

    frames = []

    for angle in angles:
        # rotate the volume
        v = rotate_on_axial_plane(vol, angle)

        # compute sagittal MIP
        mip = v.max(axis=2)

        # normalize and convert to RGB
        img = cm((mip - vol.min()) / (vol.max() - vol.min()))[..., :3]

        if mask1 is not None or mask2 is not None:
            # create RGBA overlay
            overlay = np.ones((*mip.shape, 4), dtype=np.float32)
            overlay[..., :3] = img

            for idx, m in enumerate((mask1, mask2)):
                if m is not None:
                    # rotate and project the mask
                    m_rot = rotate_on_axial_plane(m, angle)
                    m_proj = m_rot.max(axis=2) > 0

                    # blend color into overlay
                    overlay[m_proj, :3] = (
                        (1 - alpha) * overlay[m_proj, :3] +
                        alpha * np.array(colors[idx])
                    )

            frame = ax.imshow(overlay, origin="lower", aspect=spacing[0]/spacing[1], animated=True)
        else:
            frame = ax.imshow(mip, cmap="gray", origin="lower", aspect=spacing[0]/spacing[1], animated=True)

        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=100)
    ani.save("animation.gif")
    plt.close(fig)


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
    ct_datasets = ct_datasets[:-1]
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


# windowing to improve dynamic range/contrast
def apply_window(image: np.ndarray,
                 vmin: float,
                 vmax: float) -> np.ndarray:

    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")

    clipped = np.clip(image, vmin, vmax)
    scaled = ((clipped - vmin) / (vmax - vmin)) * 255.0
    return scaled.astype(np.uint8)


def visualize(base: List[np.ndarray], spacing, titles: List[str],
              masks: List[Optional[np.ndarray]] = None,):
   
    alpha = 0.30
    dz, dy, dx = spacing
    exts = ([0, dx * base[0].shape[1], 0, dy * base[0].shape[0]],
            [0, dx * base[1].shape[1], 0, dz * len(base[1])],
            [0, dy * base[2].shape[1], 0, dz * len(base[2])])

    colors = [(0.0, 1.0, 0.0),
              (1.0, 0.0, 0.0)]

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    for i, ax in enumerate(axs):
        ax.imshow(base[i], cmap='gray', extent=exts[i], origin='lower')

        if masks:
            for m_idx, m in enumerate(masks):
                if m is not None:
                    mask = m[i]
                    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                    color = colors[m_idx % len(colors)]
                    overlay[mask > 0] = [*color, alpha]
                    ax.imshow(overlay, extent=exts[i], origin='lower')

        ax.set_title(titles[i])
        ax.axis('off')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def show_midplanes(vol, spacing, mask1=None, mask2=None):
    # extract center slices in all three planes
    z, y, x = np.array(vol.shape) // 2
    base = [vol[z], vol[:, y, :], vol[:, :, x]]
    masks = None
    if mask1 is not None or mask2 is not None:
        masks = [[m[z], m[:, y, :], m[:, :, x]] if m is not None else None for m in (mask1, mask2)]
    visualize(base, spacing, [f'Axial z={z}', f'Coronal y={y}', f'Sagittal x={x}'], masks)


def show_mip_planes(vol, spacing, mask1=None, mask2=None, titles=['Axial MIP', 'Coronal MIP', 'Sagittal MIP']):
    # compute maximum intensity projections in three planes
    base = [vol.max(0), vol.max(1), vol.max(2)]
    masks = None
    if mask1 is not None or mask2 is not None:
        masks = [[m.max(0), m.max(1), m.max(2)] if m is not None else None for m in (mask1, mask2)]
    visualize(base, spacing, titles, masks)


def show_aip_planes(vol, spacing, mask1=None, mask2=None):
    # compute average intensity projections in three planes
    base = [vol.mean(0), vol.mean(1), vol.mean(2)]
    masks = None
    if mask1 is not None or mask2 is not None:
        masks = [[m.max(0), m.max(1), m.max(2)] if m is not None else None for m in (mask1, mask2)]
    visualize(base, spacing, ['Axial AIP', 'Coronal AIP', 'Sagittal AIP'], masks)


# save DICOM header to text file
def save_header(path, out_txt):
    ds = pydicom.dcmread(path)
    with open(out_txt, 'w') as f:
        f.write(str(ds))


def extract_largest_tumor_slice(tumor_mask: np.ndarray) -> tuple:
    # Find slice with largest tumor area in each plane
    z_areas = [np.sum(tumor_mask[i]) for i in range(tumor_mask.shape[0])]
    y_areas = [np.sum(tumor_mask[:,i,:]) for i in range(tumor_mask.shape[1])]
    x_areas = [np.sum(tumor_mask[:,:,i]) for i in range(tumor_mask.shape[2])]
    
    max_z = np.argmax(z_areas)
    max_y = np.argmax(y_areas)
    max_x = np.argmax(x_areas)
    
    return max_z, max_y, max_x


def show_tumor_at_indices(volume: np.ndarray, mask: np.ndarray, spacing: tuple, 
                         indices: tuple[int, int, int], title_prefix: str = "", provided_mask: np.ndarray = None):
    z_idx, y_idx, x_idx = indices

    # Extract the slices at given indices
    base = [
        volume[z_idx],          
        volume[:, y_idx, :],    
        volume[:, :, x_idx]     
    ]
    masks = []

    # If provided_mask is given, extract its slices and add to masks
    if provided_mask is not None:
        provided_mask_slices = [
            provided_mask[z_idx],
            provided_mask[:, y_idx, :],
            provided_mask[:, :, x_idx]
        ]
        masks.append(provided_mask_slices)

    # Extract corresponding mask slices
    mask_slices = [
        mask[z_idx],           
        mask[:, y_idx, :],     
        mask[:, :, x_idx]      
    ]

    masks.append(mask_slices)

    titles = [
        f'{title_prefix}Axial (z={z_idx})',
        f'{title_prefix}Coronal (y={y_idx})',
        f'{title_prefix}Sagittal (x={x_idx})'
    ]

    # Visualize
    visualize(base, spacing, titles, masks)

def get_tumor_bbox_and_centroid(mask: np.ndarray):
    # Get indices of the mask (non-zero values)
    coords = np.argwhere(mask)

    # Bounding box: min and max along each axis
    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    # Centroid: mean coordinate of the tumor
    centroid = coords.mean(axis=0)

    bbox = {
        'z': (min_z, max_z),
        'y': (min_y, max_y),
        'x': (min_x, max_x)
    }

    centroid = tuple(centroid)

    return bbox, centroid

def segment_tumor_watershed(volume: np.ndarray, initial_mask: np.ndarray, margin: int = 1) -> np.ndarray:
    # Get bounding box from initial mask
    bbox, _ = get_tumor_bbox_and_centroid(initial_mask)
    z0, z1 = bbox['z']
    y0, y1 = bbox['y']
    x0, x1 = bbox['x']
    
    # Add margin to bounding box
    z0, z1 = max(0, z0-margin), min(volume.shape[0], z1+margin)
    y0, y1 = max(0, y0-margin), min(volume.shape[1], y1+margin)
    x0, x1 = max(0, x0-margin), min(volume.shape[2], x1+margin)
    
    # Extract subvolume
    subvol = volume[z0:z1, y0:y1, x0:x1]

    # Create binary mask using Otsu threshold
    otsu_thresh = threshold_otsu(subvol)
    mask = subvol > otsu_thresh

    # # Use percentile mask
    # mask = subvol > np.percentile(subvol, 50)
    
    # Compute distance transform for watershed
    distance = distance_transform_edt(mask)

    # Find markers for watershed using local maxima
    local_max = peak_local_max(distance, labels=mask, footprint=np.ones((3,3,3)), exclude_border=False)
    markers = np.zeros_like(subvol, dtype=np.int32)
    for i, (z, y, x) in enumerate(local_max):
        markers[z, y, x] = i + 1
    
    # Apply watershed transform
    labels_ws = watershed(-distance, markers, mask=mask)
    
    # Keep only largest connected component
    labeled_array, _ = label(labels_ws > 0)
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # ignore background
    largest = np.argmax(sizes)
    segmented = (labeled_array == largest)
    
    # Place segmentation back in full volume
    full_mask = np.zeros_like(volume, dtype=bool)
    full_mask[z0:z1, y0:y1, x0:x1] = segmented
    return full_mask


def segment_tumor_region_growing(volume: np.ndarray, initial_mask: np.ndarray, 
                               threshold: float = 0.1, margin: int = 1) -> np.ndarray:
    # Get bounding box from initial mask
    bbox, _ = get_tumor_bbox_and_centroid(initial_mask)
    z0, z1 = bbox['z']
    y0, y1 = bbox['y']
    x0, x1 = bbox['x']
    
    # Add margin to bounding box
    z0, z1 = max(0, z0-margin), min(volume.shape[0], z1+margin)
    y0, y1 = max(0, y0-margin), min(volume.shape[1], y1+margin)
    x0, x1 = max(0, x0-margin), min(volume.shape[2], x1+margin)
    
    # Extract and normalize subvolume
    subvol = volume[z0:z1, y0:y1, x0:x1].astype(np.float32)
    subvol = (subvol - subvol.min()) / (subvol.max() - subvol.min())
    
    # Find seed point (brightest point)
    max_coords = np.unravel_index(np.argmax(subvol), subvol.shape)
    cz, cy, cx = max_coords
    seed_intensity = subvol[cz, cy, cx]
    
    # Initialize segmentation
    segmented = np.zeros_like(subvol, dtype=bool)
    queue = [(cz, cy, cx)]
    visited = set()
    
    # Region growing
    while queue:
        z, y, x = queue.pop(0)
        if (z, y, x) not in visited:
            visited.add((z, y, x))
            if (0 <= z < subvol.shape[0] and 
                0 <= y < subvol.shape[1] and 
                0 <= x < subvol.shape[2]):
                # Check if voxel is similar to seed
                if abs(subvol[z,y,x] - seed_intensity) < threshold:
                    segmented[z,y,x] = True
                    neighbors = [
                        (z-1,y,x), (z+1,y,x),
                        (z,y-1,x), (z,y+1,x),
                        (z,y,x-1), (z,y,x+1)
                    ]
                    queue.extend(
                        (nz,ny,nx) for nz,ny,nx in neighbors 
                        if (nz,ny,nx) not in visited
                    )
    
    # Place segmentation back in full volume
    full_mask = np.zeros_like(volume, dtype=bool)
    full_mask[z0:z1, y0:y1, x0:x1] = segmented
    return full_mask

def evaluate_segmentation(true_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    y_true = true_mask.flatten()
    y_pred = pred_mask.flatten()
    return {
        "Dice": f1_score(y_true, y_pred),
        "Jaccard": jaccard_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

if __name__ == "__main__":

    # generate headers for each type of file
    save_header(next(REF_DIR.glob("*.dcm")), "header_ct.txt")
    save_header(TUMOR_PATH, "header_tumor.txt")
    save_header(LIVER_PATH, "header_liver.txt")

    # load the reference directory
    volume, spacing_mm = load_dicom_volume(REF_DIR)
    volume = apply_window(volume, vmin=-300, vmax=500)

    # verify single aquisition
    is_single = verify_single_acquisition(REF_DIR)

    # generate masks
    mask_volume_liver = build_mask_volume(REF_DIR, LIVER_PATH)
    mask_volume_tumor = build_mask_volume(REF_DIR, TUMOR_PATH)

    animate_rotating_MIP_planes(volume, spacing_mm, mask_volume_liver, mask_volume_tumor)

    # visualize
    show_midplanes(volume, spacing_mm, mask_volume_liver, mask_volume_tumor)
    show_mip_planes(volume, spacing_mm, mask_volume_liver, mask_volume_tumor)
    show_aip_planes(volume, spacing_mm, mask_volume_liver, mask_volume_tumor)

    # Get indices of largest tumor slices
    max_z, max_y, max_x = extract_largest_tumor_slice(mask_volume_tumor)
    
    # Watershed
    watershed_mask = segment_tumor_watershed(volume, mask_volume_tumor)

    # Region growing
    region_grown_mask = segment_tumor_region_growing(volume, mask_volume_tumor, threshold=0.60)
    
    # show_mip_planes(volume, spacing_mm, None, mask_volume_tumor)
    show_mip_planes(volume, spacing_mm, mask_volume_tumor, watershed_mask, ['Watershed Axial MIP', 'Watershed Coronal MIP', 'Watershed Sagittal MIP'])
    show_mip_planes(volume, spacing_mm, mask_volume_tumor, region_grown_mask, ['Region Growing Axial MIP', 'Region Growing Coronal MIP', 'Region Growing Sagittal MIP'])

    show_tumor_at_indices(volume, watershed_mask, spacing_mm, (max_z, max_y, max_x), "Watershed ", provided_mask=mask_volume_tumor)
    show_tumor_at_indices(volume, region_grown_mask, spacing_mm, (max_z, max_y, max_x), "Region Growing ", provided_mask=mask_volume_tumor)

    metrics_ws = evaluate_segmentation(mask_volume_tumor, watershed_mask)
    print("Watershed metrics:", metrics_ws)

    metrics_rg = evaluate_segmentation(mask_volume_tumor, region_grown_mask)
    print("Region Growing metrics:", metrics_rg)

    # Print volumes of original tumor mask
    tumor_volume = np.sum(mask_volume_tumor)
    print(f"\nOriginal tumor volume: {tumor_volume} voxels")

    # Segment using watershed and print volume
    watershed_volume = np.sum(watershed_mask)
    print(f"Watershed segmentation volume: {watershed_volume} voxels")

    # Region growing and print volume
    region_growing_volume = np.sum(region_grown_mask)
    print(f"Region growing segmentation volume: {region_growing_volume} voxels")

    # Print volume differences
    ws_diff = ((watershed_volume - tumor_volume) / tumor_volume) * 100
    rg_diff = ((region_growing_volume - tumor_volume) / tumor_volume) * 100
    print(f"\nVolume differences from ground truth:")
    print(f"Watershed: {ws_diff:.1f}%")
    print(f"Region Growing: {rg_diff:.1f}%")