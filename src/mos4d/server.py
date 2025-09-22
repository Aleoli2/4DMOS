# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino, Ignacio Vizzo, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
from collections import deque
from pathlib import Path
from typing import Optional,TypedDict, cast

import numpy as np
import torch
import zmq
from kiss_icp.pipeline import OdometryPipeline
from tqdm.auto import trange

from mos4d.config import load_config
from mos4d.metrics import get_confusion_matrix
from mos4d.mos4d import MOS4DNet
from mos4d.odometry import Odometry
from mos4d.utils.pipeline_results import MOSPipelineResults
from mos4d.utils.save import KITTIWriter, StubWriter
from mos4d.utils.visualizer import MOS4DVisualizer, StubVisualizer

from dataclasses import asdict, dataclass

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor as _rebuild_tensor
import argparse
import time


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Converts a PyTorch dtype to its string representation."""

    return str(dtype).replace("torch.", "")


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    """Converts a string representation of a PyTorch dtype back to its corresponding dtype object."""

    return getattr(torch, dtype)


class SerializedCudaRebuildMetadata(TypedDict):
    """TypedDict representing a serializable version of CUDA tensor rebuild metadata."""

    dtype: str
    tensor_size: tuple[int, ...]
    tensor_stride: tuple[int, ...]
    tensor_offset: int
    storage_device: int
    storage_handle: str
    storage_size_bytes: int
    storage_offset_bytes: int
    requires_grad: bool
    ref_counter_handle: str
    ref_counter_offset: int
    event_handle: str
    event_sync_required: bool


@dataclass
class CudaRebuildMetadata:
    """Data class representing the metadata required to rebuild a CUDA tensor."""

    dtype: torch.dtype
    tensor_size: torch.Size
    tensor_stride: tuple[int, ...]
    tensor_offset: int
    storage_device: int
    storage_handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    requires_grad: bool
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool

    @classmethod
    def from_serialized_dict(cls, metadata: SerializedCudaRebuildMetadata) -> "CudaRebuildMetadata":
        """Creates a `CudaRebuildMetadata` instance from a serialized dictionary."""

        return cls(
            dtype=str_to_torch_dtype(metadata["dtype"]),
            tensor_size=torch.Size(metadata["tensor_size"]),
            tensor_stride=tuple(metadata["tensor_stride"]),
            tensor_offset=metadata["tensor_offset"],
            storage_device=metadata["storage_device"],
            storage_handle=bytes.fromhex(metadata["storage_handle"]),
            storage_size_bytes=metadata["storage_size_bytes"],
            storage_offset_bytes=metadata["storage_offset_bytes"],
            requires_grad=metadata["requires_grad"],
            ref_counter_handle=bytes.fromhex(metadata["ref_counter_handle"]),
            ref_counter_offset=metadata["ref_counter_offset"],
            event_handle=bytes.fromhex(metadata["event_handle"]),
            event_sync_required=metadata["event_sync_required"],
        )

    def to_serialized_dict(self) -> SerializedCudaRebuildMetadata:
        """Converts this `CudaRebuildMetadata` instance into a serializable dictionary."""

        metadata = asdict(self)
        metadata["dtype"] = torch_dtype_to_str(self.dtype)
        metadata["tensor_size"] = tuple(self.tensor_size)
        metadata["storage_handle"] = self.storage_handle.hex()
        metadata["ref_counter_handle"] = self.ref_counter_handle.hex()
        metadata["event_handle"] = self.event_handle.hex()
        return cast(SerializedCudaRebuildMetadata, metadata)


def share_cuda_tensor(tensor: torch.Tensor) -> CudaRebuildMetadata:
    """Shares the CUDA memory of a tensor and generates `CudaRebuildMetadata` for rebuilding the tensor later.

    Args:
        tensor (torch.Tensor): The CUDA tensor to share.

    Returns:
        CudaRebuildMetadata: Metadata required to rebuild the shared CUDA tensor.
    """

    storage = tensor._typed_storage()
    (
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()

    return CudaRebuildMetadata(
        dtype=tensor.dtype,
        tensor_size=tensor.size(),
        tensor_stride=tensor.stride(),
        tensor_offset=tensor.storage_offset(),
        storage_device=device,
        storage_handle=handle,
        storage_size_bytes=storage_size_bytes,
        storage_offset_bytes=storage_offset_bytes,
        requires_grad=tensor.requires_grad,
        ref_counter_handle=ref_counter_handle,
        ref_counter_offset=ref_counter_offset,
        event_handle=event_handle,
        event_sync_required=event_sync_required,
    )


def rebuild_cuda_tensor(metadata: CudaRebuildMetadata) -> torch.Tensor:
    """Rebuilds a CUDA tensor from the provided `CudaRebuildMetadata`.

    Args:
        metadata (CudaRebuildMetadata): The metadata required to rebuild the tensor.

    Returns:
        torch.Tensor: The rebuilt CUDA tensor.
    """

    return _rebuild_tensor(
        tensor_cls=torch.Tensor,
        tensor_size=metadata.tensor_size,
        tensor_stride=metadata.tensor_stride,
        tensor_offset=metadata.tensor_offset,
        storage_cls=torch.TypedStorage,
        dtype=metadata.dtype,
        storage_device=metadata.storage_device,
        storage_handle=metadata.storage_handle,
        storage_size_bytes=metadata.storage_size_bytes,
        storage_offset_bytes=metadata.storage_offset_bytes,
        requires_grad=metadata.requires_grad,
        ref_counter_handle=metadata.ref_counter_handle,
        ref_counter_offset=metadata.ref_counter_offset,
        event_handle=metadata.event_handle,
        event_sync_required=metadata.event_sync_required,
    )


def prob_to_log_odds(prob):
    odds = np.divide(prob, 1 - prob + 1e-10)
    log_odds = np.log(odds)
    return log_odds


class MOS4DAgent(OdometryPipeline):
    def __init__(
        self,
        weights: Path,
        config: Optional[Path] = None,
    ):
        # Config and output dir
        self.config = load_config(config)
        self.results_dir = None

        # Pipeline
        state_dict = {
            k.replace("model.", ""): v for k, v in torch.load(weights)["state_dict"].items()
        }
        state_dict = {k.replace("mos.", ""): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}

        self.model = MOS4DNet(self.config.mos.voxel_size_mos)
        self.model.load_state_dict(state_dict)
        self.model.cuda().eval().freeze()

        self.odometry = Odometry(self.config.data, self.config.odometry)
        self.buffer = deque(maxlen=self.config.mos.delay_mos)
        self.dict_logits = {}
        self.dict_gt_labels = {}
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind("tcp://*:5555")


    def _preprocess(self, points, min_range, max_range):
        ranges = np.linalg.norm(points - self.odometry.current_location(), axis=1)
        mask = ranges <= max_range if max_range > 0 else np.ones_like(ranges, dtype=bool)
        mask = np.logical_and(mask, ranges >= min_range)
        return mask

    def run(self):
        scan_index = 0
        while True:
            cuda_rebuild_metadata = cast(CudaRebuildMetadata, self.sock.recv_pyobj())
            local_scan = rebuild_cuda_tensor(cuda_rebuild_metadata)

            timestamps = time.clock_gettime_ns(time.CLOCK_REALTIME)
            scan_points = self.odometry.register_points(local_scan, timestamps, scan_index)
            scan_index += 1

            min_range_mos = self.config.mos.min_range_mos
            max_range_mos = self.config.mos.max_range_mos
            scan_mask = self._preprocess(scan_points, min_range_mos, max_range_mos)
            scan_points = torch.tensor(scan_points[scan_mask], dtype=torch.float32, device="cuda")

            self.buffer.append(
                torch.hstack(
                    [
                        scan_points,
                        scan_index
                        * torch.ones(len(scan_points)).reshape(-1, 1).type_as(scan_points),
                    ]
                )
            )

            past_point_clouds = torch.vstack(list(self.buffer))
            pred_logits = self.model.predict(past_point_clouds)

            # Detach, move to CPU
            pred_logits = pred_logits.detach().cpu().numpy().astype(np.float64)
            scan_points = scan_points.cpu().numpy().astype(np.float64)
            past_point_clouds = past_point_clouds.cpu().numpy().astype(np.float64)
            torch.cuda.empty_cache()

            # Fuse predictions in binary Bayes filter
            for past_scan_index in np.unique(past_point_clouds[:, -1]):
                mask_past_scan = past_point_clouds[:, -1] == past_scan_index
                scan_logits = pred_logits[mask_past_scan]

                if past_scan_index not in self.dict_logits.keys():
                    self.dict_logits[past_scan_index] = scan_logits
                else:
                    self.dict_logits[past_scan_index] += scan_logits
                    self.dict_logits[past_scan_index] -= prob_to_log_odds(self.config.mos.prior)

            pred_labels = self.model.to_label(pred_logits)

            mask_scan = past_point_clouds[:, -1] == scan_index
            dynamic_prediction = np.where(pred_labels[mask_scan] == 1)[0]
            self.sock.send_pyobj(share_cuda_tensor(dynamic_prediction))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOS4D Server")
    parser.add_argument("--weights", "-w", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--config", "-c", type=Path, required=False, help="Path to config file")
    args = parser.parse_args()
    agent = MOS4DAgent(args.weights, args.config)
    agent.run()

