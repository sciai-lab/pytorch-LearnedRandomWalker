from randomwalker.RandomWalkerModule import RandomWalker
import torch
import matplotlib.pyplot as plt
import numpy as np
from randomwalker.randomwalker_loss_utils import NHomogeneousBatchLoss
import os

if not os.path.exists('results'):
    os.makedirs('results')


def create_simple_image(size, batch=1):
    """
    returns a simple target image with two and a seeds mask
    """
    sizex, sizey = size

    target = torch.zeros(batch, 1, sizex, sizey).long()
    target[:, :, :, sizey//2:] = 1

    seeds = torch.zeros(batch, sizex, sizey).long()
    seeds[:, 3 * sizex//4, 5] = 1
    seeds[:, sizex//4, sizey - 5] = 2

    return seeds, target


def make_summary_plot(it, output, net_output, seeds, target):
    """
        This function create and save a summary figure
    """
    f, axarr = plt.subplots(2, 2, figsize=(8, 9.5))
    f.suptitle("RW summary, Iteration: " + repr(it))

    axarr[0, 0].set_title("Ground Truth Image")
    axarr[0, 0].imshow(target[0, 0].detach().numpy(), alpha=0.8, vmin=-3, cmap="prism_r")
    seeds_listx, seeds_listy = np.where(seeds[0].data != 0)
    axarr[0, 0].scatter(seeds_listy[1],
                        seeds_listx[1], c="w")
    axarr[0, 0].scatter(seeds_listy[0],
                        seeds_listx[0], c="k")
    axarr[0, 0].axis("off")

    axarr[0, 1].set_title("LRW output (white seed)")
    axarr[0, 1].imshow(output[0][0, 0].detach().numpy(), cmap="gray")
    axarr[0, 1].axis("off")

    axarr[1, 0].set_title("Vertical Diffusivities")
    axarr[1, 0].imshow(net_output[0, 0].detach().numpy(), cmap="gray", vmax=1)
    axarr[1, 0].axis("off")

    axarr[1, 1].set_title("Horizontal Diffusivities")
    axarr[1, 1].imshow(net_output[0, 1].detach().numpy(), cmap="gray", vmax=1)
    axarr[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("./results/%04i.png"%it)
    plt.close()


if __name__ == '__main__':
    # Init parameters
    batch_size = 1
    iterations = 60
    size = (60, 59)

    # Init the random walker modules
    rw = RandomWalker(1000, max_backprop=True)

    # Load data and init
    seeds, target = create_simple_image(size)
    diffusivities = torch.zeros(batch_size, 2, size[0], size[1], requires_grad=True)

    # Init optimizer
    optimizer = torch.optim.Adam([diffusivities], lr=0.5)

    # Loss has to been wrapped in order to work with random walker algorithm
    loss = NHomogeneousBatchLoss(torch.nn.NLLLoss)

    # Main overfit loop
    for it in range(iterations + 1):
        optimizer.zero_grad()

        # Diffusivities must be positive
        net_output = torch.sigmoid(diffusivities)

        # Random walker
        output = rw(net_output, seeds)

        # Loss and diffusivities update
        output_log = [torch.log(o) for o in output]
        l = loss(output_log, target)
        l.backward(retain_graph=True)
        optimizer.step()

        # Summary
        if it % 5 == 0:
            print("Iteration: ", it, " Loss: ", l.item())
            make_summary_plot(it, output, net_output, seeds, target)
