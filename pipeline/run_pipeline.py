
import re
import json
import shutil
import subprocess
from pathlib import Path
import argparse

EV_PER_HARTREE = 27.2114  # constant for conversion if needed

def run_xtb(xyz_path, charge, solvent, gfn=2):
    workdir = Path.cwd() / "xtb_work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    in_xyz = workdir / "input.xyz"
    shutil.copyfile(xyz_path, in_xyz)

    cmd = [
        "xtb", str(in_xyz),
        "--gfn", str(gfn),
        "--ohess",
        "--iterations", "1000",
        "--chrg", str(charge),
        "--json",
    ]

    if solvent and solvent.lower() != "none":
        cmd += ["--alpb", solvent.lower()]

    proc = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (workdir / "xtb.out").write_text(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError("xTB failed. See xtb.out.")

    opt_xyz = workdir / "xtbopt.xyz"
    json_file = workdir / "xtbout.json"
    charges_file = workdir / "charges"

    if not opt_xyz.exists():
        raise FileNotFoundError("xtbopt.xyz missing")

    return workdir, opt_xyz, json_file, charges_file

def extract_gap(json_path, xtb_out_path=None):
    if json_path and json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            for k in ["HOMO-LUMO gap/eV", "HOMO-LUMO gap (eV)", "HOMO-LUMO gap"]:
                if k in data and data[k] is not None:
                    return float(data[k])
        except:
            pass
    if xtb_out_path and xtb_out_path.exists():
        try:
            txt = xtb_out_path.read_text()
            m = re.search(r"HOMO-LUMO\s+gap\s+([\-+]?\d+(\.\d+)?)\s*eV", txt)
            if m:
                return float(m.group(1))
        except:
            pass
    return None

def extract_all_frequencies(xtb_out_path):
    if not xtb_out_path.exists():
        return []
    txt = xtb_out_path.read_text()
    freqs = []
    if "projected vibrational frequencies" in txt.lower():
        for line in txt.splitlines():
            if line.strip().lower().startswith("eigval"):
                freqs.extend([float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", line)])
    if not freqs:
        blocks = re.findall(r"frequencies\s+\[cm\^-1\]\s*\n((?:\s*[-+]?\d+(?:\.\d+)?\s*)+)", txt, flags=re.IGNORECASE)
        if blocks:
            freqs = [float(x) for x in blocks[0].split()]
    return freqs

def extract_frequencies(xtb_out_path, n=12):
    allf = extract_all_frequencies(xtb_out_path)
    return allf[:n] if allf else []

def write_information_json(out_path, central_atom, vertex_atoms, solvent, gap, reference="", dg=None):
    info = {
        "Guest - Central Atom": central_atom,
        "Guest - Vertex Atoms": vertex_atoms,
        "HOMO-LUMO Gap": gap,
        "Reference": reference,
        "Solvent": solvent,
        "dG": dg
    }
    out_path.write_text(json.dumps(info, indent=4))









def main():
    import argparse
    import shutil
    import subprocess
    from pathlib import Path
    import json
    import re

    EV_PER_HARTREE = 27.2114

    ap = argparse.ArgumentParser(description="Generate xTB-based ML input files")
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--solvent", required=True)
    ap.add_argument("--charge", type=int, required=True)
    ap.add_argument("--central-atom", type=int, required=True)
    ap.add_argument("--vertex-atoms", type=str, required=True)
    ap.add_argument("--reference", type=str, default="")
    ap.add_argument("--dg", type=float, default=400)
    ap.add_argument("--imag-threshold", type=float, default=-35.0,
                    help="Flag REOPT_NEEDED if any final frequency < this (cm^-1). Default: -35.0")
    ap.add_argument("--minor-tol", type=float, default=-5.0,
                    help="Ignore tiny numerical negatives above this (cm^-1). Default: -5.0")
    args = ap.parse_args()

    xyz_stem = Path(args.xyz).stem
    run_once_marker = Path("/tmp") / f"run_once_{xyz_stem}.done"
    if run_once_marker.exists():
        print(f"[run-once] Detected duplicate invocation for '{xyz_stem}'. Skipping this run.")
        return

    start_xyz = Path(args.xyz)


    work_base = Path.cwd() / "xtb_work"
    final_workdir = None
    final_opt_xyz = None
    final_json_file = None
    final_charges_file = None
    final_freqs = []
    success = False
    failure_reason = None

    results_root = Path("/inputs/xtb_results")
    results_root.mkdir(exist_ok=True)
    compound_folder = results_root / xyz_stem
    compound_folder.mkdir(parents=True, exist_ok=True)

    # ---- Initial xTB run
    try:
        workdir, opt_xyz, json_file, charges_file = run_xtb(start_xyz, args.charge, args.solvent)
        final_workdir, final_opt_xyz, final_json_file, final_charges_file = workdir, opt_xyz, json_file, charges_file
        final_freqs = extract_frequencies(workdir / "xtb.out", n=12)
        print(f"Initial calculation frequencies: {final_freqs}")
        success = True
    except Exception as e:
        failure_reason = f"Initial xTB failed: {e}"
        print(f"ERROR: {failure_reason}")
        success = False
        if work_base.exists():
            for p in work_base.iterdir():
                try:
                    shutil.copy2(p, compound_folder / p.name)
                except Exception:
                    pass

    # ---- Retry logic (up to 2 restarts)
    max_retries = 2
    retry_count = 0

    def has_significant_negative(freqs):
        return bool(freqs) and any(f < args.minor_tol for f in freqs)

    while success and has_significant_negative(final_freqs) and retry_count < max_retries:
        xtbhess_path = final_workdir / "xtbhess.xyz" if final_workdir else None
        if not xtbhess_path or not xtbhess_path.exists():

            break

        retry_count += 1

        safe_xtbhess = Path.cwd() / f"xtbhess_restart_{retry_count}.xyz"
        shutil.copyfile(xtbhess_path, safe_xtbhess)

        try:
            workdir, opt_xyz, json_file, charges_file = run_xtb(safe_xtbhess, args.charge, args.solvent)
            final_workdir, final_opt_xyz, final_json_file, final_charges_file = workdir, opt_xyz, json_file, charges_file
            final_freqs = extract_frequencies(workdir / "xtb.out", n=12)
            print(f"Recalculation #{retry_count} frequencies: {final_freqs}")
        except Exception as e:
            print(f"ERROR: Restart #{retry_count} failed: {e}")
            break

    print(f"Final frequencies used: {final_freqs}")

    # ---- Archive working files
    if final_workdir and final_workdir.exists():
        for p in final_workdir.iterdir():
            try:
                shutil.copy2(p, compound_folder / p.name)
            except Exception:
                pass

    # ---- Classification
    if not (success and final_opt_xyz and final_charges_file):
        job_status = "FAILED"
        status_detail = "(xTB did not converge)"
    else:
        neg_freqs = [f for f in final_freqs if f < args.minor_tol]
        min_imag = min(neg_freqs) if neg_freqs else 0.0
        if neg_freqs and min_imag < args.imag_threshold:
            job_status = "REOPT_NEEDED"
            status_detail = f"(min_imag={min_imag:.2f} cm^-1, threshold={args.imag_threshold:.1f})"
        else:
            job_status = "SUCCESS"
            status_detail = "(no significant imaginary modes)"


    print(f"JOB STATUS: {job_status} {status_detail}")
    (compound_folder / "status.txt").write_text(f"JOB STATUS: {job_status} {status_detail}\n")

    if job_status == "FAILED":
        run_once_marker.write_text("failed\n")
        return

    # ---- Write ML input files
    gap_ev = extract_gap(final_json_file, final_workdir / "xtb.out" if final_workdir else None)
    gap = gap_ev / EV_PER_HARTREE if gap_ev is not None else None

    vertex_atoms = []
    if args.vertex_atoms.strip().lower() not in {"nan", "none", ""}:
        try:
            vertex_atoms = [int(v.strip()) for v in args.vertex_atoms.split(",") if v.strip()]
        except ValueError:
            print("WARNING: Invalid vertex-atoms format.")

    idx = 0
    shutil.copyfile(final_opt_xyz, Path(f"input-{idx:04}-geometry.xyz"))
    shutil.copyfile(final_charges_file, Path(f"input-{idx:04}-charges"))

    info_json = Path(f"input-{idx:04}-information.json")
    write_information_json(info_json, args.central_atom, vertex_atoms, args.solvent, gap,
                           reference=args.reference, dg=args.dg)

    try:

        result = subprocess.run(
            ["python", "-m", "data.create_input_dataset"],
            cwd="/app",
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)

        result = subprocess.run(["python", "-m", "pipeline.calculate_input_affinity"],
                                cwd="/app", capture_output=True, text=True, check=True)
        print(result.stdout)
    except Exception as e:
        print(f"WARNING: Post-processing failed: {e}")
    finally:
        run_once_marker.write_text("done\n")





if __name__ == "__main__":
    main()
