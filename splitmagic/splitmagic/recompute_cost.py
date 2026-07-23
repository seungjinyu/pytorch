from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch.fx as fx


@dataclass(frozen=True)
class RecomputeCostEstimate:
    total_ms: float
    profiled_nodes: tuple[str, ...]
    missing_nodes: tuple[str, ...]


class RecomputeCostDB:
    """
    FX node profiling CSV를 읽어 recomputation 비용을 조회한다.

    기본 정책:
    - concurrency=1
    - avg_ms 사용
    """

    def __init__(
        self,
        csv_path: str | Path,
        concurrency: int = 1,
        metric: str = "avg_ms",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.concurrency = concurrency
        self.metric = metric

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Recompute profile CSV not found: {self.csv_path}"
            )

        self._cost_by_node: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            required_columns = {
                "node_name",
                "concurrency",
                self.metric,
            }

            if reader.fieldnames is None:
                raise ValueError(
                    f"CSV has no header: {self.csv_path}"
                )

            missing_columns = required_columns - set(reader.fieldnames)

            if missing_columns:
                raise ValueError(
                    f"CSV missing columns: {sorted(missing_columns)}"
                )

            for row in reader:
                try:
                    row_concurrency = int(row["concurrency"])
                except (TypeError, ValueError):
                    continue

                if row_concurrency != self.concurrency:
                    continue

                node_name = row["node_name"].strip()

                if not node_name:
                    continue

                try:
                    cost_ms = float(row[self.metric])
                except (TypeError, ValueError):
                    continue

                self._cost_by_node[node_name] = cost_ms

        if not self._cost_by_node:
            raise ValueError(
                "No profiling rows loaded for "
                f"concurrency={self.concurrency}, metric={self.metric}"
            )

        print(
            "[RecomputeCostDB][LOAD] "
            f"path={self.csv_path} "
            f"concurrency={self.concurrency} "
            f"metric={self.metric} "
            f"nodes={len(self._cost_by_node)}"
        )

    def get_cost_ms(
        self,
        node_name: str,
    ) -> Optional[float]:
        return self._cost_by_node.get(node_name)

    def estimate_node_names(
        self,
        node_names: Iterable[str],
    ) -> RecomputeCostEstimate:
        total_ms = 0.0
        profiled_nodes: list[str] = []
        missing_nodes: list[str] = []

        for node_name in node_names:
            cost_ms = self.get_cost_ms(node_name)

            if cost_ms is None:
                missing_nodes.append(node_name)
                continue

            total_ms += cost_ms
            profiled_nodes.append(node_name)

        return RecomputeCostEstimate(
            total_ms=total_ms,
            profiled_nodes=tuple(profiled_nodes),
            missing_nodes=tuple(missing_nodes),
        )

    def estimate_fx_nodes(
        self,
        nodes: Iterable[fx.Node],
    ) -> RecomputeCostEstimate:
        executable_nodes = []

        for node in nodes:
            if node.op in {
                "placeholder",
                "get_attr",
                "output",
            }:
                continue

            executable_nodes.append(node.name)

        return self.estimate_node_names(executable_nodes)

    def __len__(self) -> int:
        return len(self._cost_by_node)

    @property
    def costs(self) -> Mapping[str, float]:
        return self._cost_by_node