from __future__ import annotations

from dataclasses import dataclass
from shutil import which

from ...errors import UnsupportedError
from ...hardware.host import Host
from ..lowering import linear_block_plan
from ...core.operator import Operator, Parallel
from ...schedule.plan import (
    CodePlan,
    DecompositionKind,
    FusionKind,
    LoweringKind,
    MergeKind,
    MicrokernelSpec,
    PackingKind,
    TileShape,
    WorkspaceABI,
)
from ...core.validate import ParallelCall, ValidatedCall


@dataclass(frozen=True, slots=True)
class Backend:
    backend_id: str
    architecture: str
    required_features: frozenset[str]
    vector_bytes: int
    compiler: str
    cflags: tuple[str, ...]
    sve_vector_bytes: int | None = None

    def supports(self, host: Host) -> bool:
        return (
            host.architecture == self.architecture
            and self.required_features <= host.features
            and (
                self.sve_vector_bytes is None
                or host.sve_vector_bytes == self.sve_vector_bytes
            )
        )

    def compiler_path(self) -> str:
        path = which(self.compiler)
        if path is None:
            raise UnsupportedError(f"backend {self.backend_id} requires compiler {self.compiler!r}")
        return path

    def enumerate_code_plans(
        self,
        operator: Operator,
        call: ValidatedCall,
        host: Host,
    ) -> tuple[CodePlan, ...]:
        if not self.supports(host):
            raise UnsupportedError(
                f"backend {self.backend_id} is incompatible with host {host.fingerprint[:12]}"
            )
        lanes = self.vector_bytes // 4
        plans: list[CodePlan] = []
        if isinstance(operator, Parallel):
            assert isinstance(call, ParallelCall)
            shapes = (
                (8 if lanes >= 16 else 6, 1),
                (6 if lanes >= 16 else 4, 2),
                (4 if lanes >= 16 else 2, 4),
            )
            if lanes >= 16:
                shapes = shapes + ((8, 2),)
            for tile_q, qk_vectors in shapes:
                for multiplier in (1, 2):
                    for dv_multiplier in (1, 2):
                        for packing in (PackingKind.NONE, PackingKind.K_TRANSPOSED):
                            regions = (
                                ("scratch",)
                                if packing is PackingKind.NONE
                                else ("packed_k", "scratch")
                            )
                            plans.append(
                                self._code_plan(
                                    LoweringKind.PARALLEL_BLOCKED,
                                    DecompositionKind.QUERY_BLOCKS,
                                    TileShape(
                                        tile_q,
                                        qk_vectors * lanes * multiplier,
                                        call.q.shape[3],
                                        lanes * dv_multiplier,
                                    ),
                                    MicrokernelSpec(
                                        f"qk_m{tile_q}n{qk_vectors}_pv_m{tile_q}",
                                        qk_vectors,
                                    ),
                                    WorkspaceABI(regions),
                                    packing=packing,
                                )
                            )
            if operator.row_norm.supports_split_k:
                for tile_k in (2 * lanes, 4 * lanes):
                    tile = TileShape(1, tile_k, call.q.shape[3], lanes)
                    microkernel = MicrokernelSpec("qk_m1n2_pv_m1", 2)
                    for packing in (PackingKind.NONE, PackingKind.K_TRANSPOSED):
                        regions = (
                            ("partials",)
                            if packing is PackingKind.NONE
                            else ("packed_k", "partials")
                        )
                        plans.extend((
                            self._code_plan(
                                LoweringKind.PARALLEL_SPLIT_K,
                                DecompositionKind.SPLIT_KEY,
                                tile,
                                microkernel,
                                WorkspaceABI(regions),
                                packing=packing,
                                merge=MergeKind.ROW_NORM,
                            ),
                            self._code_plan(
                                LoweringKind.PARALLEL_2D,
                                DecompositionKind.QUERY_GROUPS_KEY_PARTS,
                                tile,
                                microkernel,
                                WorkspaceABI(regions),
                                packing=packing,
                                merge=MergeKind.ROW_NORM,
                            ),
                        ))
        else:
            scan_tiles = (
                TileShape(1, 1, lanes, lanes),
                TileShape(1, 1, 2 * lanes, lanes),
                TileShape(1, 1, lanes, 2 * lanes),
            )
            for tile in scan_tiles:
                plans.append(
                    self._code_plan(
                        LoweringKind.LINEAR_SCAN,
                        DecompositionKind.STATE_HEADS,
                        tile,
                        MicrokernelSpec(f"linear_m1n{lanes}"),
                        WorkspaceABI(("scratch",)),
                        packing=PackingKind.NONE,
                    )
                )
            plans.append(
                self._code_plan(
                    LoweringKind.LINEAR_2D,
                    DecompositionKind.STATE_HEADS_DV_BLOCKS,
                    TileShape(1, 1, lanes, lanes),
                    MicrokernelSpec(f"linear_dv_m1n{lanes}"),
                    WorkspaceABI(("scratch",)),
                    packing=PackingKind.NONE,
                )
            )
            if linear_block_plan(operator) is not None:
                for block in ((2, 4, 6) if lanes >= 8 else (2, 4)):
                    plans.append(
                        self._code_plan(
                            LoweringKind.LINEAR_CHUNKED,
                            DecompositionKind.STATE_HEADS,
                            TileShape(block, block, lanes, lanes),
                            MicrokernelSpec(f"linear_chunked_m{block}n{lanes}"),
                            WorkspaceABI(("scratch",)),
                            packing=PackingKind.NONE,
                        )
                    )
        unique = {plan.identity: plan for plan in plans}
        return tuple(unique[key] for key in sorted(unique))

    def _code_plan(
        self,
        lowering: LoweringKind,
        decomposition: DecompositionKind,
        tile: TileShape,
        microkernel: MicrokernelSpec,
        workspace_abi: WorkspaceABI,
        *,
        packing: PackingKind = PackingKind.K_TRANSPOSED,
        merge: MergeKind = MergeKind.NONE,
    ) -> CodePlan:
        return CodePlan(
            backend_id=self.backend_id,
            lowering=lowering,
            microkernel=microkernel,
            tile=tile,
            vector_bytes=self.vector_bytes,
            decomposition=decomposition,
            packing=packing,
            merge=merge,
            fusion=FusionKind.PATTERN_OWNED,
            workspace_abi=workspace_abi,
        )


__all__ = ["Backend"]
