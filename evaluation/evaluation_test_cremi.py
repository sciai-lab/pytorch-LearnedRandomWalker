import h5py
import numpy as np
from cremi import Volume
from cremi.evaluation import NeuronIds


def cremi_score(gt, seg, return_all_scores=True, b_thresh=2, data_resolution=(1.0, 1.0, 1.0)):
    """compute cremi scores from np.array"""

    if len(gt.shape) == 2:
        gt = gt[None, :, :]
        seg = seg[None, :, :]
    gt_ = Volume(gt, resolution=data_resolution)
    seg_ = Volume(seg, resolution=data_resolution)

    metrics = NeuronIds(gt_, b_thresh)
    arand = metrics.adapted_rand(seg_)

    vi_s, vi_m = metrics.voi(seg_)
    # official cremi score
    cs = np.sqrt(vi_s * (vi_m + arand))

    if return_all_scores:
        return cs, vi_s, vi_m, arand
    else:
        return cs


def compute_scores(gtpath, segpath, gtdataset="targets", segdataset="segmentation"):
    # Data load
    with h5py.File(gtpath, "r") as f:
        gt = f[gtdataset][...]

    with h5py.File(segpath, "r") as f:
        segmentation = f[segdataset][...]

    # Compute 2D cremi scores
    scores = []
    for i in range(gt.shape[0]):
        scores.append(cremi_score(gt[i], segmentation[i], b_thresh=2))

    scores = np.array(scores)
    rand, vois, voim = scores[:, 3], scores[:, 1],  scores[:, 2]
    print(f"results A rand:  {np.mean(rand[:50])} pm {np.std(rand[:50])}")
    print(f"results A vois:  {np.mean(vois[:50])} pm {np.std(vois[:50])}")
    print(f"results A voim:  {np.mean(voim[:50])} pm {np.std(voim[:50])}\n")

    print(f"results B rand:  {np.mean(rand[50:100])} pm {np.std(rand[50:100])}")
    print(f"results B vois:  {np.mean(vois[50:100])} pm {np.std(vois[50:100])}")
    print(f"results B voim:  {np.mean(voim[50:100])} pm {np.std(voim[50:100])}\n")

    print(f"results C rand:  {np.mean(rand[100:])} pm {np.std(rand[100:])}")
    print(f"results C vois:  {np.mean(vois[100:])} pm {np.std(vois[100:])}")
    print(f"results C voim:  {np.mean(voim[100:])} pm {np.std(voim[100:])}\n")

    print(f"results rand:    {np.mean(rand)} pm {np.std(rand)}")
    print(f"results vois:    {np.mean(vois)} pm {np.std(vois)}")
    print(f"results voim:    {np.mean(voim)} pm {np.std(voim)}")
    return scores


if __name__ == "__main__":
    import argparse

    def _parser():
        parser = argparse.ArgumentParser(description='Run cremi valuation on 2D '
                                                     '(x/y plane, test set A[0:50]B[0:50]C[0:50]).')
        parser.add_argument('--gtpath', type=str, help='Path to the groundtruth file (only h5).',
                            required=True)
        parser.add_argument('--segpath', type=str, help='Path to the predicted segmentation file (only h5).',
                            required=True)
        parser.add_argument('--gtdataset', type=str, help='Groundtruth labels dataset.',
                            default="targets", required=False)
        parser.add_argument('--segdataset', type=str, help='Predicted labels dataset.',
                            default="segmentation", required=False)
        return parser.parse_args()

    args = _parser()
    compute_scores(args.gtpath, args.segpath, args.gtdataset, args.segdataset)
