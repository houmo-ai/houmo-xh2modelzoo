# NVIDIA GPU Performance Counter Permission

## Symptom

Running `examples_merak/llm/tools/run_graph_module_nvidia_profile.sh` with Nsight Compute can fail with:

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

This means the current user cannot access NVIDIA GPU performance counters, so `ncu` cannot generate a `.ncu-rep` report.

## Current Host Facts

- CUDA is installed under `/usr/local/cuda`
- `ncu` is available from `/usr/local/cuda/bin/ncu`
- The node-profiling workflow in this repository uses NVTX push/pop ranges around GraphModule nodes

## Recommended Permanent Fix

Ask the machine administrator to enable GPU performance-counter access for non-admin users.

Create a modprobe config file:

```bash
sudo tee /etc/modprobe.d/nvidia-profiler.conf >/dev/null <<'EOF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
```

If the machine should remain restricted to admin users only, use `1` instead of `0`:

```bash
sudo tee /etc/modprobe.d/nvidia-profiler.conf >/dev/null <<'EOF'
options nvidia NVreg_RestrictProfilingToAdminUsers=1
EOF
```

Then rebuild initrd if required by the distribution:

Debian or Ubuntu:

```bash
sudo update-initramfs -u -k all
```

RedHat or CentOS:

```bash
sudo dracut --regenerate-all -f
```

Finally reboot the machine, or reload the NVIDIA kernel modules using a maintenance window.

## Temporary Fix Without Reboot

This requires root access and will interrupt GPU users on the machine.

```bash
sudo systemctl isolate multi-user
sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
sudo systemctl isolate graphical
```

If module unload fails because devices are busy, inspect the remaining holders first:

```bash
sudo lsof /dev/nvidia*
```

## Verification

Check whether the loaded driver still requires admin-only profiling:

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

Expected meaning:

- `RmProfilingAdminOnly: 1` means only admin users can profile
- `RmProfilingAdminOnly: 0` means all users can profile

Check whether the modprobe config was included in initrd:

Debian or Ubuntu:

```bash
sudo lsinitramfs /boot/initrd.img | grep /etc/modprobe.d
```

RedHat or CentOS:

```bash
sudo lsinitrd | grep /etc/modprobe.d
```

## Short Message For Admin

```text
We need NVIDIA GPU performance-counter access enabled for Nsight Compute on this host.

Current error:
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters.

Requested permanent fix:
1. Add `/etc/modprobe.d/nvidia-profiler.conf` with:
   options nvidia NVreg_RestrictProfilingToAdminUsers=0
2. Rebuild initrd if needed:
   - Debian/Ubuntu: update-initramfs -u -k all
   - RedHat/CentOS: dracut --regenerate-all -f
3. Reboot, or reload the NVIDIA kernel modules during a maintenance window.

Verification:
grep RmProfilingAdminOnly /proc/driver/nvidia/params

Reference:
https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

## Notes

- Running `ncu` with `sudo` or with `CAP_SYS_ADMIN` can work around the restriction, but that is not a durable fix for shared development workflows.
- For containers, NVIDIA's guidance says host-side access must be enabled, or the container must be started with `--cap-add=SYS_ADMIN` by an admin user.
