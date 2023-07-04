import os
from typing import Optional

import numpy as np
import torch
import torch.utils.data
import tqdm
from hyfi.composer import BaseConfig
from hyfi.utils.logging import LOGGING
from pydantic import ConfigDict, Field
from skimage.io import imread, imsave

from forger.ui.brush import GanBrushOptions, PaintEngineFactory, PaintingHelper
from forger.ui.library import BrushLibrary
from forger.util import img_proc
from forger.viz import style_transfer

logger = LOGGING.getLogger(__name__)


class ForgerStylize(BaseConfig):
    _config_group_ = "forger"
    _config_name_ = "stylize"

    model_dir: str = Field("models", description="Path to model directory")
    input_dir: str = Field("inputs", description="Path to input directory")
    output_dir: str = Field("outputs", description="Path to output directory")
    tmp_dir: str = Field("tmp", description="Path to tmp directory")

    model_name: str = Field("style2", description="Model name")
    gan_checkpoint: str = Field("snapshot.pkl", description="Name of GAN checkpoint")
    encoder_checkpoint: Optional[str] = Field(
        None, description="Path to encoder checkpoint"
    )
    output_file_prefix: str = Field(..., description="Desired output file prefix")
    geom_image: Optional[str] = Field(None, description="Path to geometry image")
    stitching_mode: str = Field("all", description="Which patches to paint")
    feature_blending_level: int = Field(0, description="Feature blending level")
    library: str = Field("rand100", description="Which library to use")
    style_id: str = Field(..., description="Style ID")
    style_id2: Optional[str] = Field(None, description="Second style ID")
    style_blend_alpha: float = Field(0.5, description="Style blend alpha")
    crop_margin: int = Field(10, description="Crop margin")
    render_mode: str = Field("clear", description="Render mode")
    no_uvs_mapping: bool = Field(False, description="Disable UVS mapping.")
    color_mode: Optional[str] = Field(None, description="Color mode")
    on_white: bool = Field(False, description="On white")
    debug: bool = Field(False, description="Debug mode")

    model_config = ConfigDict(protected_namespaces=())

    @property
    def gan_checkpoint_path(self):
        return os.path.join(
            self.model_dir, "neube", self.model_name, self.gan_checkpoint
        )

    @property
    def encoder_checkpoint_path(self):
        if self.encoder_checkpoint is None:
            return None
        return os.path.join(self.model_dir, self.encoder_checkpoint)

    @property
    def library_path(self):
        return os.path.join(
            self.model_dir, "neube", self.model_name, "brush_libs", self.library
        )

    @property
    def geom_image_path(self):
        if self.geom_image is None:
            return None
        return os.path.join(self.input_dir, self.geom_image)

    @property
    def debug_file_path(self):
        return os.path.join(self.tmp_dir, "geo.png")

    def output_file_path(self, style_name):
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(
            self.output_dir,
            f"{self.output_file_prefix}_{self.render_mode}_{str(style_name)}.png",
        )

    def paint(self):
        device = torch.device(0)

        engine = PaintEngineFactory.create(
            encoder_checkpoint=self.encoder_checkpoint_path,
            gan_checkpoint=self.gan_checkpoint_path,
            device=device,
        )
        library = BrushLibrary.from_arg(self.library_path, z_dim=engine.G.z_dim)

        brush_options = GanBrushOptions()
        brush_options.enable_uvs_mapping = not self.no_uvs_mapping
        brush_options.debug = False
        if self.color_mode is not None:
            ForgerStylize.set_colors(
                self.color_mode,
                library,
                engine.uvs_mapper,
                self.style_id,
                self.style_id2,
                brush_options,
            )
        if self.style_id2 is None:
            library.set_style(self.style_id, brush_options)
        else:
            library.set_interpolated_style(
                self.style_id, self.style_id2, self.style_blend_alpha, brush_options
            )

        patch_width = engine.G.img_resolution
        geom = ForgerStylize._read_any_geo(self.geom_image_path)
        orig_geo_shape = geom.shape
        geom = ForgerStylize.pad_geo(geom, self.crop_margin)

        # Gets crops and pads geometry
        stitching_crops, geom = style_transfer.generate_stitching_crops(
            geom,
            patch_width,
            mode=self.stitching_mode,
            overlap_margin=self.crop_margin * 2,
        )
        result = np.zeros((geom.shape[0], geom.shape[1], 4), dtype=np.uint8)
        if self.debug:
            imsave(
                self.debug_file_path,
                ForgerStylize.visualize_crops(geom, stitching_crops),
            )
            raise RuntimeError("stop")

        helper = PaintingHelper(engine)
        helper.make_new_canvas(
            result.shape[0],
            result.shape[1],
            feature_blending=self.feature_blending_level,
        )
        helper.set_render_mode(self.render_mode)

        with torch.no_grad():
            for i in tqdm.tqdm(range(len(stitching_crops)), total=len(stitching_crops)):
                y = stitching_crops[i][0]
                x = stitching_crops[i][1]
                brush_options.set_position(x, y)
                geom_patch = (
                    255 - geom[y : y + patch_width, x : x + patch_width, :]
                )  # why is this reverse??

                res, _, meta = helper.render_stroke(
                    geom_patch,
                    None,
                    brush_options,
                    meta={"x": x, "y": y, "crop_margin": self.crop_margin},
                )

                res_y = meta["y"]
                res_x = meta["x"]
                res_height = res.shape[0]
                res_width = res.shape[1]
                result[res_y : res_y + res_height, res_x : res_x + res_width, :] = res

        if self.on_white:
            alpha = result[..., 3:].astype(np.float32) / 255
            result = result[..., :3].astype(np.float32) * alpha + 255 * (1 - alpha)
            result[..., 3:] = 255
            result = result.clip(0, 255).astype(np.uint8)

        result = result[
            self.crop_margin : self.crop_margin + orig_geo_shape[0],
            self.crop_margin : self.crop_margin + orig_geo_shape[1],
            :,
        ]

        style_name = self.style_id
        if self.style_id2 is not None:
            style_name += "_%0.1f%s" % (self.style_blend_alpha, self.style_id2)
        output_file = self.output_file_path(style_name)
        imsave(output_file, result)
        logger.info(f"Saved result to: {output_file}")

    @staticmethod
    def _read_any_geo(fname):
        """
        @param fname: filename of an image
        @return: 1 x 1 x W' x H' torch float32 image
        """
        img = torch.from_numpy(imread(fname)).to(torch.float32)

        if len(img.shape) == 2:
            img = img.unsqueeze(-1)

        if img.shape[2] == 3:
            img = img[..., :3].mean(dim=2).unsqueeze(-1)
        elif img.shape[2] == 4:
            mean = img[..., :3].mean(dim=2)
            alpha = img[..., 3] / 255
            img = (mean * alpha + 255 * (1 - alpha)).unsqueeze(-1)

        mn = torch.min(img)
        if mn > 0:
            img = img - mn

        mx = torch.max(img)
        if 0 < mx < 255:
            img = img * (255.0 / mx.item())

        img = img.to(torch.uint8).numpy()
        img = (
            img_proc.threshold_img(img, to_float=False).astype(np.float32) * 255
        ).astype(np.uint8)
        return img

    @staticmethod
    def pad_geo(geo, crop_margin):
        geo_padded = (
            np.ones(
                (geo.shape[0] + crop_margin, geo.shape[1] + crop_margin, geo.shape[2]),
                dtype=np.uint8,
            )
            * 255
        )
        geo_padded[crop_margin:, crop_margin:, :] = geo
        return geo_padded

    @staticmethod
    def set_colors(color_mode, library, mapper, style_id1, style_id2, brush_options):
        if color_mode in ["1", "2"]:
            opts = GanBrushOptions()
            if color_mode == "1":
                library.set_style(style_id1, opts)
            elif color_mode == "2":
                library.set_style(style_id2, opts)
            colors = mapper.get_colors_raw(opts)
            print(colors)
            brush_options.set_color(0, colors[0, :, 0] / 2 + 0.5)
            brush_options.set_color(1, colors[0, :, 1] / 2 + 0.5)
        else:
            color_specs = color_mode.split(";")
            for i, cspec in enumerate(color_specs):
                if len(cspec) > 0:
                    rgb = [int(x) for x in cspec.split(",")]
                    assert len(rgb) == 3
                    brush_options.set_color(
                        i, torch.tensor(rgb, dtype=torch.float32) / 255.0
                    )
                    logger.info(f"Set color {i} to {rgb}")

    @staticmethod
    def visualize_crops(geom, crops):
        result = np.concatenate([geom, geom, geom], axis=2)
        for crop in crops:
            y = crop[0]
            x = crop[1]
            width = crop[2]
            result[y : y + width, x, :] = 0
            result[y : y + width, x + width - 1, :] = 0
            result[y, x : x + width, :] = 0
            result[y + width - 1, x : x + width, :] = 0
            result[y : y + width, x, 0] = 255
            result[y : y + width, x + width - 1, 0] = 255
            result[y, x : x + width, 0] = 255
            result[y + width - 1, x : x + width, 0] = 255
        return result
