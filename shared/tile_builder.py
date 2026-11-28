"""Shared Gaussian-first tile-list construction for Warp renderers.

The builder consumes projected screen-space records.  Projection is deliberately left to
the caller because the standalone renderer receives NumPy projected records, while the
trainer projects trainable Gaussians on the device.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from time import perf_counter

import warp as wp

# Opt-in stage profiling, off by default. Attributing GPU time to a stage requires a device
# synchronise around it, which serialises work that normally overlaps -- so leaving this on would
# both slow training and distort the timings being measured. Enable with
# WARP_3DGS_PROFILE_TILES=1 and read `builder.stage_times` (seconds) after a build.
PROFILE_TILES = os.environ.get("WARP_3DGS_PROFILE_TILES", "") not in ("", "0", "false")


@wp.kernel(enable_backward=False)
def count_projected_tile_pairs(
    centres: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    supports: wp.array[wp.float32],
    count: int,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
    tiles_y: int,
    tile_bounds: wp.array[wp.vec4i],
    pair_counts: wp.array[wp.int32],
):
    """Find the tile rectangle touched by each projected Gaussian."""
    splat = wp.tid()
    pair_counts[splat] = 0
    tile_bounds[splat] = wp.vec4i(0, -1, 0, -1)

    if splat < count:
        centre = centres[splat]
        conic = conics[splat]
        support = supports[splat]
        determinant = conic[0] * conic[2] - conic[1] * conic[1]
        if determinant > 1.0e-12 and support > 0.0:
            cov_xx = conic[2] / determinant
            cov_yy = conic[0] / determinant
            radius_x = wp.sqrt(wp.max(support * cov_xx, 0.0))
            radius_y = wp.sqrt(wp.max(support * cov_yy, 0.0))

            if (
                wp.isfinite(centre[0])
                and wp.isfinite(centre[1])
                and wp.isfinite(radius_x)
                and wp.isfinite(radius_y)
            ):
                min_px = int(wp.floor(centre[0] - radius_x))
                max_px = int(wp.ceil(centre[0] + radius_x)) - 1
                min_py = int(wp.floor(centre[1] - radius_y))
                max_py = int(wp.ceil(centre[1] + radius_y)) - 1

                if max_px >= 0 and max_py >= 0 and min_px < width and min_py < height:
                    min_px = wp.max(min_px, 0)
                    max_px = wp.min(max_px, width - 1)
                    min_py = wp.max(min_py, 0)
                    max_py = wp.min(max_py, height - 1)
                    min_tx = wp.max(min_px // tile_size, 0)
                    max_tx = wp.min(max_px // tile_size, tiles_x - 1)
                    min_ty = wp.max(min_py // tile_size, 0)
                    max_ty = wp.min(max_py // tile_size, tiles_y - 1)

                    if min_tx <= max_tx and min_ty <= max_ty:
                        tile_bounds[splat] = wp.vec4i(min_tx, max_tx, min_ty, max_ty)
                        pair_counts[splat] = (max_tx - min_tx + 1) * (
                            max_ty - min_ty + 1
                        )


@wp.kernel(enable_backward=False)
def emit_projected_tile_pairs(
    tile_bounds: wp.array[wp.vec4i],
    depths: wp.array[wp.float32],
    pair_counts: wp.array[wp.int32],
    pair_prefix: wp.array[wp.int32],
    group_ids: wp.array[wp.int32],
    splat_ids: wp.array[wp.uint32],
    tiles_per_view: int,
    tiles_x: int,
    depth_keys: wp.array[wp.float32],
    packed_pairs: wp.array[wp.uint64],
):
    """Write one packed (tile-group, splat) record per tile overlap.

    This function is called per projected 2D Gaussian.

    `tile_bounds`: This is the bound on the ranges of tiles that are affected by the thread's Gaussian.
    `depths`: This is the array of **keys** to the actual depth value of the projected 2D Gaussians retained for rendering.
    `pair_counts`: This is the number of tiles that this Gaussian affects.
    `pair_prefix`: This is the prefix sum that records this Gaussian affects.
    `group_ids`: This is the id of the camera views (can have multiple) that this Gaussian is related to.
    `splat_ids`: This is the id of the Gaussian splats (currently it is just identity mapping),
                 but with multiple views, this will not be true (one Gaussian would correspond to multiple projected Gaussians.)
    `tiles_per_view`: This is just a constant that decides the number of tiles each camera view has (assuming equal resolution).
    `tiles_x`: This is just how many tiles along the horizontal for each view.
    `depth_keys`: This is the depth for each projectwd Gaussian.
                  Again since each camera has different extrinsic,
                  each view needs a separate depth keys hence this design.
    `packed_pairs`: This is the output that is being written as a pair of the tile and the projected splat index.
    """
    item = wp.tid()
    count = pair_counts[item]
    if count > 0:
        # output records to location to write to.
        # Note that prefix is the prefix sum including the current splat.
        # By subtracting the current count, we know what is the count up the previous.
        bounds = tile_bounds[item]
        output = pair_prefix[item] - count
        group_base = group_ids[item] * tiles_per_view
        # Note that this piece of code is designed to work for multiple views of the same scene.
        # The Gaussians are projected already and each view would have different projected Gaussians.
        for tile_y in range(bounds[2], bounds[3] + 1):
            for tile_x in range(bounds[0], bounds[1] + 1):
                group = group_base + tile_y * tiles_x + tile_x
                depth_keys[output] = depths[item]
                packed_pairs[output] = (wp.uint64(group) << wp.uint64(32)) | wp.uint64(
                    splat_ids[item]
                )
                output += 1


@wp.kernel(enable_backward=False)
def unpack_group_keys(
    packed_pairs: wp.array[wp.uint64],
    group_keys: wp.array[wp.uint32],
):
    """This function is used extract the tile id information and sort it so that later sorting again becomes:
    ```bash
    (splat_close, ... , splat_further), (splat_close, ... , splat_further), ... , (splat_close, ... , splat_further)
    ^tile id 1                          ^tile id 2                                ^tile id nx * ny
    ```
    """
    item = wp.tid()
    group_keys[item] = wp.uint32(packed_pairs[item] >> wp.uint64(32))


@wp.kernel(enable_backward=False)
def count_sorted_groups(
    group_keys: wp.array[wp.uint32],
    group_counts: wp.array[wp.int32],
):
    item = wp.tid()
    # wp.atomic_add(dst, idx, val)
    # `dst` is the pointer to the array
    # `idx` is the location to be added
    # `val` is the value to be added
    # Therefore, the following lines mean add 1 to the tile corresponding to this record.
    # Atomic to avoid race conditions (multiple records would correspond to the same tile).
    wp.atomic_add(group_counts, int(group_keys[item]), 1)


@wp.kernel(enable_backward=False)
def copy_pair_count(
    pair_prefix: wp.array[wp.int32],
    item_count: int,
    pair_count: wp.array[wp.int32],
):
    if item_count > 0:
        pair_count[0] = pair_prefix[item_count - 1]
    else:
        pair_count[0] = 0


class GaussianFirstTileBuilder:
    """Persistent buffers and common pipeline for projected Gaussian tile records."""

    def __init__(
        self,
        max_items,
        max_groups,
        width,
        height,
        tile_size,
        tile_pair_capacity,
        device,
    ):
        self.max_items = int(max_items)
        self.max_groups = int(max_groups)
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(tile_size)
        self.tiles_x = (self.width + self.tile_size - 1) // self.tile_size
        self.tiles_y = (self.height + self.tile_size - 1) // self.tile_size
        self.tiles_per_view = self.tiles_x * self.tiles_y
        self.tile_pair_capacity = int(tile_pair_capacity)
        self.device = wp.get_device(device)

        self.tile_bounds = wp.zeros(self.max_items, dtype=wp.vec4i, device=self.device)
        self.pair_counts = wp.zeros(self.max_items, dtype=wp.int32, device=self.device)
        self.pair_prefix = wp.zeros(self.max_items, dtype=wp.int32, device=self.device)
        self.pair_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.depth_keys = wp.zeros(
            2 * self.tile_pair_capacity, dtype=wp.float32, device=self.device
        )
        self.packed_pairs = wp.zeros(
            2 * self.tile_pair_capacity, dtype=wp.uint64, device=self.device
        )
        self.group_keys = wp.zeros(
            2 * self.tile_pair_capacity, dtype=wp.uint32, device=self.device
        )  # This contains the id link between the record orders and the corresponding tile.
        self.group_counts = wp.zeros(
            self.max_groups, dtype=wp.int32, device=self.device
        )
        self.tile_offsets = wp.zeros(
            self.max_groups + 1, dtype=wp.int32, device=self.device
        )

    def build(
        self,
        centres,
        conics,
        supports,
        depths,
        group_ids,
        splat_ids,
        item_count,
        group_count,
        width=None,
        height=None,
    ):
        """Build and sort records for projected items.

        `group_ids` identifies the view for each item.  Group zero is the only group needed
        by the standalone renderer; the training workspace uses one group per view and
        packs the local tile id into the upper half of each pair value.
        """
        if not 0 <= item_count <= self.max_items:
            raise ValueError("Projected item count exceeds tile-builder capacity.")
        if not 1 <= group_count <= self.max_groups:
            raise ValueError("Tile-group count exceeds tile-builder capacity.")
        render_width = self.width if width is None else int(width)
        render_height = self.height if height is None else int(height)
        self.stage_times = {}

        @contextmanager
        def stage(name):
            if not PROFILE_TILES:
                yield
                return
            wp.synchronize_device(self.device)
            started = perf_counter()
            yield
            wp.synchronize_device(self.device)
            self.stage_times[name] = (
                self.stage_times.get(name, 0.0) + perf_counter() - started
            )

        with stage("1_count_pairs"):
            wp.launch(
                count_projected_tile_pairs,
                dim=item_count,
                inputs=[
                    centres,
                    conics,
                    supports,
                    item_count,
                    render_width,
                    render_height,
                    self.tile_size,
                    self.tiles_x,
                    self.tiles_y,
                ],
                outputs=[self.tile_bounds, self.pair_counts],
                device=self.device,
            )
        with stage("2_scan_pair_counts"):
            wp.utils.array_scan(
                self.pair_counts[:item_count],
                self.pair_prefix[:item_count],
                inclusive=True,
            )
            wp.launch(
                copy_pair_count,
                dim=1,
                inputs=[self.pair_prefix, item_count],
                outputs=[self.pair_count],
                device=self.device,
            )
        with stage("3_readback_pair_count"):
            pair_count = int(self.pair_count.numpy()[0])
        if pair_count > self.tile_pair_capacity:
            raise RuntimeError(
                f"Tile-pair capacity {self.tile_pair_capacity:,} is too small for "
                f"{pair_count:,} splat-to-tile records. Increase --tile-pair-capacity."
            )

        self.group_counts.zero_()
        self.tile_offsets.zero_()
        if pair_count == 0:
            wp.synchronize_device(self.device)
            return self.tile_offsets, self.packed_pairs, 0

        with stage("4_emit_pairs"):
            wp.launch(
                emit_projected_tile_pairs,
                dim=item_count,
                inputs=[
                    self.tile_bounds,
                    depths,
                    self.pair_counts,
                    self.pair_prefix,
                    group_ids,
                    splat_ids,
                    self.tiles_per_view,
                    self.tiles_x,
                ],
                outputs=[self.depth_keys, self.packed_pairs],
                device=self.device,
            )
        # First sort by exact depth. The second stable sort groups records while
        # preserving near-to-far order inside each group.
        with stage("5_sort_by_depth"):
            wp.utils.radix_sort_pairs(self.depth_keys, self.packed_pairs, pair_count)
        with stage("6_unpack_group_keys"):
            wp.launch(
                unpack_group_keys,
                dim=pair_count,
                inputs=[self.packed_pairs],
                outputs=[self.group_keys],
                device=self.device,
            )
        with stage("7_sort_by_group"):
            wp.utils.radix_sort_pairs(
                self.group_keys,
                self.packed_pairs,
                pair_count,
                end_bit=max(1, (group_count - 1).bit_length()),
            )
        with stage("8_count_groups"):
            wp.launch(
                count_sorted_groups,
                dim=pair_count,
                inputs=[self.group_keys],
                outputs=[self.group_counts],
                device=self.device,
            )
        with stage("9_scan_offsets"):
            wp.utils.array_scan(
                self.group_counts[:group_count],
                self.tile_offsets[1 : group_count + 1],
                inclusive=True,
            )
        with stage("10_final_sync"):
            wp.synchronize_device(self.device)
        return self.tile_offsets, self.packed_pairs, pair_count
