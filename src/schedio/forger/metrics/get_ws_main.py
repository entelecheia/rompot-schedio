# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import logging
import os
import time

import forger.metrics.util
import forger.util.logging
import forger.viz.visualize as visualize
import numpy as np
import torch
import torch.utils.data
from forger.util.logging import log_tensor
from forger.util.torch_data import (
    get_image_data_iterator,
    get_image_data_iterator_from_dataset,
)
from skimage.io import imsave
from thirdparty.stylegan2_ada_pytorch.torch_utils.misc import InfiniteSampler
from thirdparty.stylegan2_ada_pytorch.training.dataset import ImageFolderDataset

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    aparser.add_argument('--encoder_checkpoint', action='store', type=str, required=False)
    aparser.add_argument('--output_file', action='store', type=str, required=True)
    aparser.add_argument('--style_seeds', action='store', type=str, required=True,
                         help='If int, will create this many random styles. '
                              'If csv ints, will use these seeds. '
                              'If file, will load seeds from file.')
    aparser.add_argument('--seed', action='store', type=int, default=None)
    aparser.add_argument('--overwrite', action='store_true')
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()

    forger.util.logging.default_log_setup(args.log_level)
    device = torch.device(0)

    if args.output_file == 'default':
        args.output_file = os.path.join(
            forger.viz.visualize.get_default_eval_directory(args.gan_checkpoint), 'ws.txt')
        logger.warning(f'Using default output directory: {args.output_file}')

    if os.path.isfile(args.output_file):
        if args.overwrite:
            logger.warning(f'Output file is being overwritten: {args.output_file}')
        else:
            raise RuntimeError(f'Cannot use output file, already exists: {args.output_file}')
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    random_state = forger.metrics.util.RandomState(args.seed)
    style_seeds = forger.metrics.util.style_seeds_from_flag(args.style_seeds, args.gan_checkpoint, random_state)

    generator = forger.metrics.util.PaintStrokeGenerator.create(
        encoder_checkpoint=args.encoder_checkpoint,
        gan_checkpoint=args.gan_checkpoint,
        device=device,
        batch_size=1,
        random_state=random_state)
    mapping = generator.engine.G.mapping

    with open(args.output_file, 'wb') as ofile:
        for idx, seed in enumerate(style_seeds):
            if (idx % 100) == 0:
                logger.info('Processing {} / {} ({})'.format(idx, len(style_seeds), seed))
            style_z = generator.get_random_style(seed)
            style_w = mapping(style_z, None)
            style_w = style_w[:, 0, ...].squeeze()
            ofile.write(style_w.detach().to(torch.float64).cpu().numpy().tobytes())
    print(f'Done: {args.output_file}')


