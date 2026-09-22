#!/usr/bin/env bash
# Model runtime helpers for generic local verification jobs.
# Sourced by jobs/run_verification.slurm and scripts/submit_model_ablation.sh.
# Does NOT download models, submit jobs, or run inference by itself.

# Canonical registry keys (aliases resolve here).
canonicalize_model_key() {
    local raw="${1:?model key required}"
    case "${raw}" in
        qwen|qwen25_vl_7b|qwen2_5_vl_7b|qwen2_5_vl) echo "qwen2_5_vl" ;;
        internvl|internvl3_8b|internvl3) echo "internvl3" ;;
        phi4|phi4_multimodal) echo "phi4_multimodal" ;;
        glm46v_flash|glm_4_6v_flash) echo "glm_4_6v_flash" ;;
        llava) echo "llava" ;;
        gemma) echo "gemma" ;;
        *)
            echo "ERROR: unsupported model key: ${raw}" >&2
            echo "Supported: qwen2_5_vl, phi4_multimodal, glm_4_6v_flash, internvl3, llava, gemma" >&2
            return 1
            ;;
    esac
}

# Map A1..A5 → frozen ablation folder names (must match src/paths.py).
condition_dir_name() {
    case "${1:?}" in
        A1) echo "A1_overlay_only" ;;
        A2) echo "A2_overlay_confidence" ;;
        A3) echo "A3_overlay_confidence_geometry" ;;
        A4) echo "A4_overlay_crop_confidence" ;;
        A5) echo "A5_crop_only" ;;
        *)
            echo "ERROR: CONDITION must be A1|A2|A3|A4|A5 (got ${1})" >&2
            return 1
            ;;
    esac
}

# Isolated envs recovered from successful historical Slurm scripts.
# Empty string → use the submitting shell / default cluster python.
model_venv_path() {
    case "$(canonicalize_model_key "$1")" in
        phi4_multimodal) echo "${PHI_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-phi4}" ;;
        glm_4_6v_flash) echo "${GLM_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-glm46v}" ;;
        *) echo "" ;;
    esac
}

# Default batch size per model (historical: Phi/GLM/InternVL often 1; Qwen/LLaVA 4).
model_default_batch_size() {
    case "$(canonicalize_model_key "$1")" in
        phi4_multimodal|glm_4_6v_flash|internvl3) echo "1" ;;
        *) echo "4" ;;
    esac
}

# Local checkpoints used by completed / qualification jobs (never download).
model_default_checkpoint() {
    case "$(canonicalize_model_key "$1")" in
        qwen2_5_vl) echo "/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct" ;;
        phi4_multimodal) echo "/deac/csc/yangGrp/luoz23/models/Phi-4-multimodal-instruct" ;;
        glm_4_6v_flash) echo "/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash" ;;
        internvl3) echo "/deac/csc/yangGrp/luoz23/models/InternVL3-8B-Instruct" ;;
        llava) echo "/deac/csc/yangGrp/luoz23/models/llava_onevision" ;;
        gemma) echo "/deac/csc/yangGrp/luoz23/models/gemma-3-12b-it" ;;
        *) echo "" ;;
    esac
}

# Default walltime hint by limit (Slurm --time still set by submitter).
suggest_time_limit() {
    local limit="${1:-1000}"
    if (( limit <= 20 )); then
        echo "04:00:00"
    elif (( limit <= 1000 )); then
        echo "12:00:00"
    else
        echo "24:00:00"
    fi
}

activate_model_environment() {
    local model_key
    model_key="$(canonicalize_model_key "${1:?}")"
    local venv
    venv="$(model_venv_path "${model_key}")"
    local project_dir="${2:?project dir required}"

    if [[ -n "${venv}" ]]; then
        # Isolated Phi/GLM envs (recovered from historical Slurm scripts).
        module load apps/python/3.11.8 2>/dev/null || true
        if [[ ! -d "${venv}" ]]; then
            echo "ERROR: missing model venv: ${venv}" >&2
            return 1
        fi
        # shellcheck disable=SC1091
        source "${venv}/bin/activate"
        export PYTHONNOUSERSITE=1
        unset PYTHONPATH || true
        export PATH="${venv}/bin:${PATH}"
        export PYTHONPATH="${project_dir}"
        case "${model_key}" in
            phi4_multimodal)
                case "$(which python)" in
                    */envs/wild-palm-phi4/*) ;;
                    *)
                        echo "ERROR: python is not from wild-palm-phi4: $(which python)" >&2
                        return 1
                        ;;
                esac
                ;;
            glm_4_6v_flash)
                case "$(which python)" in
                    */envs/wild-palm-glm46v/*) ;;
                    *)
                        echo "ERROR: python is not from wild-palm-glm46v: $(which python)" >&2
                        return 1
                        ;;
                esac
                ;;
        esac
    else
        # Default-cluster models (Qwen / LLaVA / Gemma / InternVL): mirror historical
        # jobs/run_qwen_ablation.slurm — do NOT module-load a bare Python that lacks
        # transformers. Use the submitting shell's python (/usr/bin/python on DEAC).
        export PYTHONPATH="${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
        if ! python -c 'import transformers' 2>/dev/null; then
            echo "ERROR: default python lacks transformers: $(which python)" >&2
            echo "Historical Qwen jobs used /usr/bin/python with cluster site-packages." >&2
            return 1
        fi
        echo "Default python: $(which python)"
        python -c 'import transformers, torch; print("transformers", transformers.__version__); print("torch", torch.__version__)'
    fi
}

# Refuse writes into known completed experiment trees (historical reproducibility).
refuse_protected_results_path() {
    local results_dir="${1:?}"
    case "${results_dir}" in
        */outputs/verification/qwen/20260706_2214/*|\
        */outputs/verification/qwen/20260708_0020/*|\
        */outputs/evaluation/qwen/20260706_2214/*|\
        */outputs/evaluation/qwen/20260708_0020/*|\
        */outputs/verification/glm_4_6v_flash/20260913_*|\
        */outputs/verification/glm_4_6v_flash/20260914_*|\
        */outputs/verification/glm_4_6v_flash/20260919_*|\
        */outputs/evaluation/glm_4_6v_flash/20260913_*|\
        */outputs/evaluation/glm_4_6v_flash/20260914_*|\
        */outputs/evaluation/glm_4_6v_flash/20260919_*|\
        */outputs/verification/phi4_multimodal/20260919_1524_*|\
        */outputs/verification/phi4_multimodal/20260920_2339_*|\
        */outputs/verification/phi4_multimodal/20260921_*|\
        */outputs/evaluation/phi4_multimodal/20260919_1524_*|\
        */outputs/evaluation/phi4_multimodal/20260920_2339_*|\
        */outputs/evaluation/phi4_multimodal/20260921_*|\
        */outputs/verification/internvl3/20260909_*|\
        */outputs/verification/llava/20260719_1734/*|\
        */outputs/verification/gemma/20260802_1702/*)
            echo "ERROR: refusing protected historical path: ${results_dir}" >&2
            return 1
            ;;
    esac
    return 0
}
