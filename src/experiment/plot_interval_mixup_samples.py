import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.fabric import setup_fabric
from method.utils import mixup_data

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def experiment(config: DictConfig) -> None:
    """
    Generate `no_hypercubes` MixUp hypercubes, sample `no_points_within_hypercube` points
    within each with noise increasing from 0 to epsilon*lam.
    Plot all samples in a single grid and save as one image.
    """
    log.info("Preparing dataset")
    cl_dataset = instantiate(config.dataset)
    task_datasets = cl_dataset.prepare_tasks(os.getenv("DATA_DIR"))

    log.info("Setting up Fabric")
    fabric = setup_fabric(config)

    alpha = config.exp.mixup_alpha
    total_hypercubes = config.exp.no_hypercubes
    points_per_cube = config.exp.no_points_within_hypercube
    saved = 0

    save_dir = os.path.join(config.exp.log_dir, "mixup_interval_samples")
    os.makedirs(save_dir, exist_ok=True)

    all_samples = []

    for task_id, dataset in enumerate(task_datasets):
        if saved >= total_hypercubes:
            break

        log.info(f"Processing Task {task_id}")

        # Load test inputs and targets
        inputs = dataset.get_test_inputs()
        targets = dataset.get_test_outputs()
        x = dataset.input_to_torch_tensor(inputs, fabric.device, mode="inference")
        y = dataset.output_to_torch_tensor(targets, fabric.device, mode="inference")
        y = y.max(dim=1)[1]

        # Apply MixUp
        mixed_x, _, _, lam = mixup_data(x, y, alpha=alpha)
        eps_transformed = abs(2 * lam - 1.0) * config.exp.epsilon
        epsilons = torch.linspace(0, eps_transformed, steps=points_per_cube, device=mixed_x.device)

        remaining = total_hypercubes - saved
        num_to_save = min(remaining, mixed_x.shape[0])

        for i in range(num_to_save):
            center = mixed_x[i]  # shape (C, H, W)

            if center.shape[0] not in [1,3]:
                center = center.permute(2, 0, 1)

            noise = torch.rand((points_per_cube,) + center.shape, device=center.device) * 2 - 1
            noise = noise * epsilons.view(-1, 1, 1, 1)  # scale noise by epsilons

            samples = center.unsqueeze(0) + noise  # (points_per_cube, C, H, W)
            samples_np = samples.detach().cpu().numpy()

            all_samples.append(samples_np)

        saved += num_to_save
        log.info(f"Collected {num_to_save} hypercubes from Task {task_id}")

    all_samples_np = np.concatenate(all_samples, axis=0)  # shape (total_hypercubes*points_per_cube, C, H, W)
    total_samples = all_samples_np.shape[0]

    rows = total_hypercubes
    cols = points_per_cube

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx >= total_samples:
            ax.axis('off')
            continue

        img = all_samples_np[idx]

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        else:
            raise ValueError(f"Unexpected channel size: {img.shape[0]}")

        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.axis('off')

        if idx % cols == 0:
            ax.set_ylabel(f"Hypercube {idx // cols + 1}", fontsize=16)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "interval_mixup_samples.png")
    plt.savefig(out_path)
    plt.close(fig)

    log.info(f"Saved combined grid plot with {total_samples} samples.")
