#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./export_nsys_csv.sh [ROOT]
#
# Example:
#   ./export_nsys_csv.sh ./nsys_results
#
# Optional:
#   REPORTS="nvtxsum,cudaapisum,gpukernsum,gpumemtimesum" ./export_nsys_csv.sh ./nsys_results
#   FORCE=1 ./export_nsys_csv.sh ./nsys_results
#
# Each .nsys-rep gets:
#   <experiment_dir>/nsys_csv/<report_base>_<report>.csv
#   <experiment_dir>/nsys_csv/<report_base>_export.log

ROOT="${1:-./nsys_results}"
NSYS="${NSYS:-nsys}"
FORCE="${FORCE:-0}"

# Requested reports. Unsupported reports are skipped automatically.
REPORTS="${REPORTS:-nvtxsum,cudaapisum,gpukernsum,gpumemtimesum}"

if ! command -v "$NSYS" >/dev/null 2>&1; then
    echo "[ERROR] nsys not found: $NSYS" >&2
    exit 1
fi

if [[ ! -d "$ROOT" ]]; then
    echo "[ERROR] directory not found: $ROOT" >&2
    exit 1
fi

echo "[INFO] ROOT=$ROOT"
echo "[INFO] NSYS=$("$NSYS" --version 2>/dev/null | head -n 1 || true)"
echo "[INFO] requested reports=$REPORTS"

# Detect report support for the installed Nsight Systems version.
HELP_REPORTS="$("$NSYS" stats --help-reports 2>&1 || true)"

IFS=',' read -r -a REQUESTED_REPORTS <<< "$REPORTS"
SUPPORTED_REPORTS=()

for report in "${REQUESTED_REPORTS[@]}"; do
    report="${report//[[:space:]]/}"

    [[ -z "$report" ]] && continue

    if grep -Eq "(^|[^[:alnum:]_])${report}([^[:alnum:]_]|$)" <<< "$HELP_REPORTS"; then
        SUPPORTED_REPORTS+=("$report")
    else
        echo "[WARN] unsupported report skipped: $report"
    fi
done

if (( ${#SUPPORTED_REPORTS[@]} == 0 )); then
    echo "[ERROR] none of the requested reports are supported." >&2
    echo "[INFO] Run: $NSYS stats --help-reports" >&2
    exit 1
fi

echo "[INFO] supported reports=${SUPPORTED_REPORTS[*]}"

count=0
success=0
failed=0
skipped=0

while IFS= read -r -d '' rep; do
    ((count += 1))

    experiment_dir="$(dirname "$rep")"
    filename="$(basename "$rep")"
    base="${filename%.nsys-rep}"
    out_dir="$experiment_dir/nsys_csv"
    log_file="$out_dir/${base}_export.log"

    mkdir -p "$out_dir"

    echo
    echo "============================================================"
    echo "[FILE] $rep"
    echo "[OUT ] $out_dir"
    echo "============================================================"

    file_failed=0

    {
        echo "source=$rep"
        echo "generated_at=$(date --iso-8601=seconds)"
        echo "nsys_version=$("$NSYS" --version 2>/dev/null | head -n 1 || true)"
        echo
    } > "$log_file"

    for report in "${SUPPORTED_REPORTS[@]}"; do
        out_csv="$out_dir/${base}_${report}.csv"

        if [[ -s "$out_csv" && "$FORCE" != "1" ]]; then
            echo "[SKIP] exists: $out_csv"
            echo "[SKIP] report=$report output=$out_csv" >> "$log_file"
            ((skipped += 1))
            continue
        fi

        tmp_csv="${out_csv}.tmp"

        echo "[RUN ] report=$report"

        # Without --output, CSV is emitted to stdout. This avoids
        # version-dependent automatic filename suffixes.
        if "$NSYS" stats \
            --report "$report" \
            --format csv \
            "$rep" \
            > "$tmp_csv" \
            2>> "$log_file"; then

            if [[ -s "$tmp_csv" ]]; then
                mv -f "$tmp_csv" "$out_csv"
                echo "[ OK ] $out_csv"
                echo "[ OK ] report=$report output=$out_csv" >> "$log_file"
            else
                rm -f "$tmp_csv"
                echo "[WARN] empty output: report=$report"
                echo "[WARN] empty output report=$report" >> "$log_file"
            fi
        else
            rm -f "$tmp_csv"
            echo "[FAIL] report=$report"
            echo "[FAIL] report=$report" >> "$log_file"
            file_failed=1
        fi
    done

    if (( file_failed == 0 )); then
        ((success += 1))
    else
        ((failed += 1))
    fi

done < <(find "$ROOT" -type f -name '*.nsys-rep' -print0 | sort -z)

echo
echo "========================== SUMMARY =========================="
echo "nsys-rep files : $count"
echo "successful     : $success"
echo "failed         : $failed"
echo "skipped CSVs   : $skipped"
echo "============================================================"

if (( count == 0 )); then
    echo "[WARN] no .nsys-rep files found under: $ROOT"
fi

if (( failed > 0 )); then
    exit 2
fi
