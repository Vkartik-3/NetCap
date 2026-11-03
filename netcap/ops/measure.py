"""Run a command and record how the whole process behaved: wall time to exit, peak memory, swapping.

  python -m netcap.ops.measure <result_name> -- <command ...>

Saves results/netcap/<result_name>.json. macOS (`/usr/bin/time -l`, `vm_stat`, `sysctl vm.swapusage`).
Swap-outs come from `vm_stat` and are system-wide: they count every process on the machine, so a value
of 0 shows that nothing swapped, while a non-zero value is an upper bound for this command.
If the command prints a line `NETCAP_END <epoch seconds>` just before returning, the time from that mark
to process exit (interpreter teardown) is recorded as `teardown_s`.
"""
import re
import subprocess
import sys
import time

from ..provenance import run_record, save_result


def vm_stat():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1))
    return {"swapouts": g("Swapouts"), "swapins": g("Swapins"), "pageouts": g("Pageouts")}


def swapusage():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    name, cmd = argv[0], argv[argv.index("--") + 1:]
    import os
    load0 = os.getloadavg()
    v0, s0 = vm_stat(), swapusage()
    t0 = time.time()
    p = subprocess.Popen(["/usr/bin/time", "-l", *cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    t1 = time.time()
    v1, s1 = vm_stat(), swapusage()
    sys.stdout.write(out)
    g = lambda pat: int(re.search(pat, err).group(1)) if re.search(pat, err) else None
    end = re.findall(r"NETCAP_END ([\d.]+)", out)
    load1 = os.getloadavg()
    rec = {"measured_command": " ".join(cmd), "returncode": p.returncode, "process_wall_s": t1 - t0,
           "load_average_before": load0, "load_average_after": load1,
           "timing_label": ("measured on an otherwise idle machine" if load0[0] <= 4 and load1[1] <= 4
                            else "upper bound, measured under CPU contention"),
           "teardown_s": (t1 - float(end[-1])) if end else None,
           "max_rss_bytes": g(r"(\d+)\s+maximum resident set size"),
           "peak_memory_footprint_bytes": g(r"(\d+)\s+peak memory footprint"),
           "user_s": float(re.search(r"([\d.]+) user", err).group(1)) if " user" in err else None,
           "sys_s": float(re.search(r"([\d.]+) sys", err).group(1)) if " sys" in err else None,
           "vm_stat_delta_system_wide": {k: v1[k] - v0[k] for k in v0}, "swapusage_before": s0, "swapusage_after": s1,
           "stderr_tail": err[-3000:]}
    print(save_result(name, run_record("see measured command's own result file", **rec)))
    print({k: rec[k] for k in ("returncode", "process_wall_s", "teardown_s", "peak_memory_footprint_bytes",
                               "vm_stat_delta_system_wide")})
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
