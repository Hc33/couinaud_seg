"""Load preprocessed CT arrays and restore vessel and Couinaud label maps."""
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk


def load_pickle(pkl_file):
    with open(pkl_file, "rb") as f:
        content = pickle.load(f)
    return content


def load_case(input_path, properties):
    """Load and normalize CT from an NPZ file and return vessel and Couinaud labels."""

    np_file = str(input_path)
    ndarray = np.load(np_file, "r")["data"]
    ct_ndarray = ndarray[0]
    vessel_labels = ndarray[2]
    couinaud_labels = ndarray[3]

    _mean = properties["liver_intensity_mean"]
    _std = properties["liver_intensity_std"]
    _intensity_00_5 = properties["liver_intensity_00_5"]
    _intensity_99_5 = properties["liver_intensity_99_5"]


    ct_ndarray = np.clip(ct_ndarray, a_min=_intensity_00_5, a_max=_intensity_99_5)
    ct_ndarray = (ct_ndarray - _mean) / _std

    return ct_ndarray, vessel_labels, couinaud_labels


def resample_mask_to_origin(mask_ndarray, pkl_content):
    """Restore a [1, C, Z, Y, X] tensor to a SimpleITK label image using saved crop and image geometry."""

    num_class = mask_ndarray.size(1)
    mask_size = mask_ndarray.size()[2:]
    target_size = pkl_content["cropped_size"]
    resize_ratio = target_size / mask_size
    zoomed_mask_nda = F.interpolate(mask_ndarray, scale_factor=tuple(resize_ratio), mode='trilinear')


    nonzero_z_min = pkl_content["before_crop_nonzero_region_z_min"]
    nonzero_z_max = pkl_content["before_crop_nonzero_region_z_max"] + 1
    nonzero_y_min = pkl_content["before_crop_nonzero_region_y_min"]
    nonzero_y_max = pkl_content["before_crop_nonzero_region_y_max"] + 1
    nonzero_x_min = pkl_content["before_crop_nonzero_region_x_min"]
    nonzero_x_max = pkl_content["before_crop_nonzero_region_x_max"] + 1
    size_after_resample = pkl_content["size_after_resample"]
    refill_nda = torch.zeros(tuple(np.concatenate([[1, num_class], size_after_resample])))
    refill_nda[:, :, nonzero_z_min:nonzero_z_max, nonzero_y_min:nonzero_y_max, nonzero_x_min:nonzero_x_max] = zoomed_mask_nda


    origin_size = pkl_content["original_size_of_raw_data"]
    resample_ratio = origin_size / size_after_resample
    back_nda = F.interpolate(refill_nda, scale_factor=tuple(resample_ratio), mode='trilinear')

    argmax_nda = torch.argmax(back_nda, dim=1)
    squeeze_nda = torch.squeeze(argmax_nda)

    np_nda = squeeze_nda.cpu().numpy().astype(np.uint8)


    new_image = sitk.GetImageFromArray(np_nda)
    new_image.SetDirection(pkl_content["itk_direction"])
    new_image.SetOrigin(pkl_content["itk_orgin"])
    new_image.SetSpacing(pkl_content["itk_spacing"])
    return new_image


def threshold_for_export(pred1_, pred2_):
    """Return copies of both probability tensors thresholded at 0.5."""
    pred1_, pred2_ = pred1_.clone(), pred2_.clone()
    threshold = 0.5
    pred1_[pred1_ >= threshold] = 1
    pred1_[pred1_ < threshold] = 0
    pred2_[pred2_ >= threshold] = 1
    pred2_[pred2_ < threshold] = 0
    return pred1_, pred2_


def save_predictions(probabilities, metadata, output_dir, save_probabilities=True):
    """Write vessel and Couinaud label maps in the original image geometry."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if save_probabilities:
        path = output_dir / "probabilities.npz"
        np.savez_compressed(path, vessel=probabilities[0].detach().cpu().numpy(),
                            couinaud=probabilities[1].detach().cpu().numpy())
        paths["probabilities"] = str(path)
    for name, prediction in zip(("vessel", "couinaud"), threshold_for_export(*probabilities)):
        image = resample_mask_to_origin(prediction, metadata)
        path = output_dir / (name + ".nii.gz")
        sitk.WriteImage(image, str(path))
        paths[name] = str(path)
    return paths
