"""
3D-volume evaluation for the PNG-trained 2D XAttnRes model.

For fair comparison with nnU-Net, metrics are computed on the full 3D label volume
(not on individual 2D slices). Inference is done slice-by-slice on the test set
imagesTs/, and the per-slice argmaxes are stacked back into a (Z, H, W) volume
that's saved as .nii.gz for downstream evaluation.

Per-slice preprocessing matches train.py exactly:
    HU clip -> min-max scale to [0, 255] uint8
    -> replicate to 3 channels -> Albumentations resize + ImageNet normalize
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from monai.metrics import (
    compute_dice,
    compute_iou,
    compute_hausdorff_distance,
    compute_average_surface_distance,
)

import XAttnResUnet
from dataloader import IMAGENET_MEAN, IMAGENET_STD


# AMOS22 class names — matches the nnU-Net evaluation script.
AMOS22_CLASSES = {
    1: 'Spleen',                2: 'Right_Kidney',          3: 'Left_Kidney',
    4: 'Gallbladder',           5: 'Esophagus',             6: 'Liver',
    7: 'Stomach',               8: 'Aorta',                 9: 'Inferior_Vena_Cava',
    10: 'Pancreas',             11: 'Right_Adrenal_Gland',  12: 'Left_Adrenal_Gland',
    13: 'Duodenum',             14: 'Bladder',              15: 'Prostate_Uterus',
}


# CT window — must match the values used in preprocess.py.
HU_LOWER = -150.0
HU_UPPER = 250.0


def hu_slice_to_uint8_rgb(hu_slice: np.ndarray) -> np.ndarray:
    """Same scheme as preprocess.py — produces (H, W, 3) uint8 in [0, 255]."""
    img = np.clip(hu_slice, HU_LOWER, HU_UPPER)
    img = (img - HU_LOWER) / (HU_UPPER - HU_LOWER)
    img = (img * 255.0).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def build_inference_transform(image_size: int):
    """Same chain as build_val_transform in dataloader.py."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def read_nifti_with_geometry(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img), img


def save_nifti_like(arr_zyx, ref, out_path):
    out = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    out.CopyInformation(ref)
    sitk.WriteImage(out, str(out_path))


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
def load_model(checkpoint_path, num_classes, device):
    model = XAttnResUnet.Model(n_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_volume_slicewise(model, image_volume, transform, image_size, device, batch_size=16):
    """Slice-by-slice inference, then stack back to a (Z, H, W) label volume at native size."""
    z, h, w = image_volume.shape
    pred_volume = np.zeros((z, h, w), dtype=np.uint8)

    for start in range(0, z, batch_size):
        end = min(start + batch_size, z)
        batch_imgs = []
        for si in range(start, end):
            img_rgb = hu_slice_to_uint8_rgb(image_volume[si].astype(np.float32))
            batch_imgs.append(transform(image=img_rgb)['image'])
        batch = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        logits = model(batch)
        pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for i, si in enumerate(range(start, end)):
            if (h, w) == (image_size, image_size):
                pred_volume[si] = pred[i]
            else:
                pred_volume[si] = zoom(pred[i].astype(np.float32),
                                       (h / image_size, w / image_size), order=0).astype(np.uint8)
    return pred_volume


def run_slicewise_inference_fold(checkpoint_path, images_dir, output_pred_dir,
                                 num_classes, image_size, device, batch_size):
    print("-" * 70)
    print(f"Inference with checkpoint: {checkpoint_path}")
    print("-" * 70)
    output_pred_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint_path, num_classes, device)
    transform = build_inference_transform(image_size)
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_0000.nii.gz')])

    for fname in tqdm(image_files, desc='inference'):
        case_id = fname.replace('_0000.nii.gz', '')
        out_path = output_pred_dir / f"{case_id}.nii.gz"
        if out_path.exists():
            continue
        img_arr, img_sitk = read_nifti_with_geometry(str(images_dir / fname))
        pred_arr = predict_volume_slicewise(
            model, img_arr.astype(np.float32), transform, image_size, device, batch_size,
        )
        save_nifti_like(pred_arr, img_sitk, out_path)
    print(f"Predictions saved in: {output_pred_dir}")


# --------------------------------------------------------------------------------------
# Per-case metrics
# --------------------------------------------------------------------------------------
def evaluate_metrics_for_fold(pred_dir, gt_dir, output_csv_dir, fold):
    print("-" * 70)
    print(f"Evaluating Fold {fold}...")
    print("-" * 70)
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(os.listdir(pred_dir))
    results = []

    for pred_name in pred_files:
        if not pred_name.endswith('.nii.gz'):
            continue
        case_id = pred_name.replace('.nii.gz', '')
        pred_path = pred_dir / pred_name
        gt_path = gt_dir / pred_name
        if not gt_path.exists():
            print(f"Warning: missing GT for {pred_name}, skipping.")
            continue

        pred_data = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(np.uint8)
        gt_data = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.uint8)
        case_metrics = {"Case ID": case_id}

        for class_idx, class_name in AMOS22_CLASSES.items():
            pred_bin = (pred_data == class_idx).astype(np.uint8)
            gt_bin = (gt_data == class_idx).astype(np.uint8)

            if gt_bin.sum() == 0 and pred_bin.sum() == 0:
                dice, jaccard = 1.0, 1.0
                hd95, asd = np.nan, np.nan
            elif gt_bin.sum() == 0 and pred_bin.sum() > 0:
                dice, jaccard = 0.0, 0.0
                hd95, asd = np.nan, np.nan
            else:
                pt = torch.from_numpy(pred_bin).unsqueeze(0).unsqueeze(0)
                gt = torch.from_numpy(gt_bin).unsqueeze(0).unsqueeze(0)
                dice = compute_dice(pt, gt, include_background=True).item()
                jaccard = compute_iou(pt, gt, include_background=True).item()
                try:
                    hd95 = compute_hausdorff_distance(pt, gt, include_background=True,
                                                     percentile=95).item()
                except Exception:
                    hd95 = np.nan
                try:
                    asd = compute_average_surface_distance(pt, gt, include_background=True).item()
                except Exception:
                    asd = np.nan

            case_metrics[f"Dice_{class_name}"] = dice * 100
            case_metrics[f"Jaccard_{class_name}"] = jaccard * 100
            case_metrics[f"95HD_{class_name}"] = hd95
            case_metrics[f"ASD_{class_name}"] = asd

        case_metrics["Dice_Average"] = np.nanmean(
            [case_metrics[f"Dice_{c}"] for c in AMOS22_CLASSES.values()])
        case_metrics["Jaccard_Average"] = np.nanmean(
            [case_metrics[f"Jaccard_{c}"] for c in AMOS22_CLASSES.values()])
        case_metrics["95HD_Average"] = np.nanmean(
            [case_metrics[f"95HD_{c}"] for c in AMOS22_CLASSES.values()])
        case_metrics["ASD_Average"] = np.nanmean(
            [case_metrics[f"ASD_{c}"] for c in AMOS22_CLASSES.values()])
        results.append(case_metrics)
        print(f"Fold {fold} | {case_id} | mean Dice: {case_metrics['Dice_Average']:.2f}%")

    df = pd.DataFrame(results)
    df.to_csv(output_csv_dir / f"amos22_metrics_per_case_fold{fold}.csv",
              index=False, float_format="%.2f")

    metrics_summary, fold_mean_results = [], {}
    for cls_name in list(AMOS22_CLASSES.values()) + ["Average"]:
        row, fold_mean_results[cls_name] = {"Category": cls_name}, {}
        for metric in ["Dice", "Jaccard", "95HD", "ASD"]:
            col = f"{metric}_{cls_name}"
            row[f"{metric} Mean"] = df[col].mean(skipna=True)
            row[f"{metric} Std"] = df[col].std(skipna=True)
            fold_mean_results[cls_name][metric] = df[col].mean(skipna=True)
        metrics_summary.append(row)
    pd.DataFrame(metrics_summary).to_csv(
        output_csv_dir / f"amos22_metrics_summary_fold{fold}.csv",
        index=False, float_format="%.2f")
    print(f"Fold {fold} evaluation saved to: {output_csv_dir}")
    return fold_mean_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='AMOS22 dataset root containing imagesTs/ and labelsTs/.')
    parser.add_argument('--num_classes', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ckpt_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--model_tag', type=str, default='xattnres')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4')
    parser.add_argument('--experiments_root', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = Path(args.dataset_dir)
    imagesTs = base_dir / "imagesTs"
    labelsTs = base_dir / "labelsTs"
    assert imagesTs.is_dir() and labelsTs.is_dir()

    experiments_root = (Path(args.experiments_root) if args.experiments_root
                        else base_dir / f"experiments_results_{args.model_tag}")
    experiments_root.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(',') if x.strip() != '']

    all_folds_metrics = []
    for fold in folds:
        print("\n" + "=" * 80 + f"\n FOLD {fold}\n" + "=" * 80)
        ckpt = Path(args.ckpt_dir) / f"{args.model_tag}_fold{fold}_best.pth"
        if not ckpt.exists():
            print(f"No checkpoint at {ckpt}, skipping fold.")
            continue
        pred_dir = experiments_root / f"predictions_raw_fold{fold}"
        eval_dir = experiments_root / f"evaluation_fold{fold}"
        run_slicewise_inference_fold(str(ckpt), imagesTs, pred_dir,
                                     args.num_classes, args.image_size, device, args.batch_size)
        all_folds_metrics.append(
            evaluate_metrics_for_fold(pred_dir, labelsTs, eval_dir, fold)
        )

    if not all_folds_metrics:
        print("\nNothing to summarize.")
        return

    # Cross-fold final summary
    print("\n" + "=" * 80 +
          f"\n Cross-fold stats over {len(all_folds_metrics)} fold(s)\n" + "=" * 80)
    final_summary = []
    for cls in list(AMOS22_CLASSES.values()) + ["Average"]:
        row = {"Category": cls}
        for metric in ["Dice", "Jaccard", "95HD", "ASD"]:
            vals = [d[cls][metric] for d in all_folds_metrics]
            valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if valid:
                row[f"{metric} CrossFold_Mean"] = float(np.mean(valid))
                row[f"{metric} CrossFold_Std"] = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
            else:
                row[f"{metric} CrossFold_Mean"] = np.nan
                row[f"{metric} CrossFold_Std"] = np.nan
        final_summary.append(row)

    final_csv = experiments_root / f"amos22_FINAL_{len(all_folds_metrics)}_folds_summary.csv"
    pd.DataFrame(final_summary).to_csv(final_csv, index=False, float_format="%.2f")

    out = (f"\nFinal ({len(all_folds_metrics)} folds, Mean +/- Std):\n" + "-" * 90 +
           f"\n{'Category':<22} | {'Dice (%)':<15} | {'Jaccard (%)':<15} | "
           f"{'95HD (px)':<14} | {'ASD (px)':<14}\n" + "-" * 90)
    for row in final_summary:
        cat = row["Category"]
        def _f(mk, sk):
            m, s = row[mk], row[sk]
            return f"{m:.2f} +/- {s:.2f}" if pd.notna(m) else "NaN"
        d = _f('Dice CrossFold_Mean', 'Dice CrossFold_Std')
        j = _f('Jaccard CrossFold_Mean', 'Jaccard CrossFold_Std')
        h = _f('95HD CrossFold_Mean', '95HD CrossFold_Std')
        a = _f('ASD CrossFold_Mean', 'ASD CrossFold_Std')
        if cat == "Average":
            out += "\n" + "-" * 90
        out += f"\n{cat:<22} | {d:<15} | {j:<15} | {h:<14} | {a:<14}"
    out += "\n" + "-" * 90
    print(out)
    print(f"\nFinal summary saved to: {final_csv}")


if __name__ == "__main__":
    main()