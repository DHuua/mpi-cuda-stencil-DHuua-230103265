import os
import sys
import json
import math
import time
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15
REPORT_EVERY = 100
DT_SAFETY = 0.5            # dt = DT_SAFETY * dt_max (explicit diffusion limit)
PROC_DIMS = [2, 2]
NY_LOC = GLOBAL_NY // PROC_DIMS[0]   # 256 owned rows per rank
NX_LOC = GLOBAL_NX // PROC_DIMS[1]   # 256 owned cols per rank
PAD_Y = NY_LOC + 2                   # 258
PAD_X = NX_LOC + 2                   # 258

# Message tags are named by the direction of travel of the strip.
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 0, 1, 2, 3


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf8")).hexdigest()
    return int(digest[:16], 16)


# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization

def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    # Non-periodic 2x2 Cartesian process grid
    cart_comm = comm.Create_cart(dims=PROC_DIMS, periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    row_rank, col_rank = cart_comm.Get_coords(cart_rank)
    # Shift(direction, disp) -> (source, dest); disp=+1: source = lower coord, dest = higher coord
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return {
        "comm": comm,
        "cart": cart_comm,
        "rank": cart_rank,
        "coords": (row_rank, col_rank),
        "up": nbr_up, "down": nbr_down, "left": nbr_left, "right": nbr_right,
        "github_user": github_user.strip(),
        "gpu_uuid": gpu_uuid,
        "seed": instance_seed,
    }


# TASK 2: Seeded Heterogeneous Subdomain Initialization (no global array)
def initialize_subdomains(ctx):
    """Build rank-local (258,258) u, kappa, S. Axis 0 = y (rows), axis 1 = x (cols).
    Global interior index I in 1..512 maps to coordinate I/513; I=0 and I=513 are the
    physical boundaries (u = 0). Local index i maps to global I = row_rank*256 + i."""
    row_rank, col_rank = ctx["coords"]
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)
    I = row_rank * NY_LOC + np.arange(PAD_Y)
    J = col_rank * NX_LOC + np.arange(PAD_X)
    y = I * dy
    x = J * dx
    Y, X = np.meshgrid(y, x, indexing="ij")

    # Identical on all ranks (same seed) -> globally consistent fields without communication
    rng = np.random.default_rng(ctx["seed"])
    ph1, ph2 = rng.uniform(0.0, 2.0 * math.pi, size=2)
    kappa = 0.16 + 0.08 * np.sin(2.0 * math.pi * X + ph1) * np.sin(2.0 * math.pi * Y + ph2)

    n_pulses = 3
    cx = rng.uniform(0.2, 0.8, n_pulses)
    cy = rng.uniform(0.2, 0.8, n_pulses)
    amp = rng.uniform(0.5, 1.5, n_pulses)
    sig = rng.uniform(0.05, 0.10, n_pulses)
    u = np.zeros((PAD_Y, PAD_X), dtype=np.float64)
    for k in range(n_pulses):
        u += amp[k] * np.exp(-((X - cx[k]) ** 2 + (Y - cy[k]) ** 2) / (2.0 * sig[k] ** 2))

    sx, sy = rng.uniform(0.3, 0.7, 2)
    S = 1.0e-3 * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2.0 * 0.04 ** 2))

    # Homogeneous Dirichlet on global outer edges (neighbor == PROC_NULL)
    if ctx["up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx["down"] == MPI.PROC_NULL:
        u[PAD_Y - 1, :] = 0.0
    if ctx["left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx["right"] == MPI.PROC_NULL:
        u[:, PAD_X - 1] = 0.0

    return {
        "u": np.ascontiguousarray(u),
        "kappa": np.ascontiguousarray(kappa),
        "S": np.ascontiguousarray(S),
        "dx": dx, "dy": dy,
    }


# TASK 3: GPU Halo Pack / Unpack Kernels & Pinned Asynchronous MPI Exchange
# Buffer layout (1D, float64), segment lengths: top/bottom = nx, left/right = ny
#   [0:nx] top   [nx:2nx] bottom   [2nx:2nx+ny] left   [2nx+ny:2nx+2ny] right

@cuda.jit
def pack_halos_kernel(u, send, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:                                   # top owned row (row 1)
        send[idx] = u[1, 1 + idx]
    elif idx < 2 * nx:                             # bottom owned row (row ny)
        send[idx] = u[ny, 1 + (idx - nx)]
    elif idx < 2 * nx + ny:                        # left owned column (col 1), strided
        send[idx] = u[1 + (idx - 2 * nx), 1]
    elif idx < 2 * nx + 2 * ny:                    # right owned column (col nx), strided
        send[idx] = u[1 + (idx - 2 * nx - ny), nx]


@cuda.jit
def unpack_halos_kernel(u, recv, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:                                   # top ghost row 0
        u[0, 1 + idx] = recv[idx]
    elif idx < 2 * nx:                             # bottom ghost row ny+1
        u[ny + 1, 1 + (idx - nx)] = recv[idx]
    elif idx < 2 * nx + ny:                        # left ghost column 0
        u[1 + (idx - 2 * nx), 0] = recv[idx]
    elif idx < 2 * nx + 2 * ny:                    # right ghost column nx+1
        u[1 + (idx - 2 * nx - ny), nx + 1] = recv[idx]


def _sync(stream):
    if isinstance(stream, int):
        cuda.synchronize()
    else:
        stream.synchronize()


def exchange_halos_async(ctx, st, d_u, stream):
    """Stage A (pack + async D2H into pinned buffers) and the posting half of Stage B
    (non-blocking Irecv/Isend). Returns the outstanding requests."""
    cart = ctx["cart"]
    ny, nx = NY_LOC, NX_LOC
    total = 2 * nx + 2 * ny
    tpb = 128
    blocks = (total + tpb - 1) // tpb
    pack_halos_kernel[blocks, tpb, stream](d_u, st["d_send"], ny, nx)
    st["d_send"].copy_to_host(st["h_send"], stream=stream)   # async D2H to page-locked memory
    _sync(stream)                                            # host send buffers are now ready

    hs, hr = st["h_send"], st["h_recv"]
    o_b, o_l, o_r = nx, 2 * nx, 2 * nx + ny
    e = total
    reqs = [
        # receives: segment = ghost destination, tag = direction of travel of the incoming strip
        cart.Irecv(hr[0:nx],   source=ctx["up"],    tag=TAG_DOWN),   # top ghost  <- up's bottom strip
        cart.Irecv(hr[o_b:o_l], source=ctx["down"], tag=TAG_UP),     # bottom ghost <- down's top strip
        cart.Irecv(hr[o_l:o_r], source=ctx["left"], tag=TAG_RIGHT),  # left ghost <- left's right strip
        cart.Irecv(hr[o_r:e],  source=ctx["right"], tag=TAG_LEFT),   # right ghost <- right's left strip
        # sends
        cart.Isend(hs[0:nx],    dest=ctx["up"],    tag=TAG_UP),
        cart.Isend(hs[o_b:o_l], dest=ctx["down"],  tag=TAG_DOWN),
        cart.Isend(hs[o_l:o_r], dest=ctx["left"],  tag=TAG_LEFT),
        cart.Isend(hs[o_r:e],   dest=ctx["right"], tag=TAG_RIGHT),
    ]
    return reqs


def finish_halo_exchange(ctx, st, d_u, stream, reqs):
    """Stage C (first half): Waitall, async H2D of received pinned halos, unpack into ghosts."""
    MPI.Request.Waitall(reqs)
    st["d_recv"].copy_to_device(st["h_recv"], stream=stream)
    total = 2 * NX_LOC + 2 * NY_LOC
    tpb = 128
    blocks = (total + tpb - 1) // tpb
    unpack_halos_kernel[blocks, tpb, stream](d_u, st["d_recv"], NY_LOC, NX_LOC)


# TASK 4: 2D shared-memory tiled stencil + dual-stream orchestrator
# Bounding box (r_start, r_end, c_start, c_end) is INCLUSIVE, in padded indices.

@cuda.jit
def tiled_stencil_kernel(u_in, u_out, kappa, S, r_start, r_end, c_start, c_end,
                         dt, inv_dy2, inv_dx2, lam):
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty

    # Center cell (only cells the bounding box can possibly read are loaded)
    if i <= r_end + 1 and j <= c_end + 1:
        smem[ty + 1, tx + 1] = u_in[i, j]
    # 1-cell halo apron loaded by edge threads
    if ty == 0 and j <= c_end + 1:
        smem[0, tx + 1] = u_in[i - 1, j]
    if ty == TILE_DIM - 1 and i + 1 <= r_end + 1 and j <= c_end + 1:
        smem[TILE_DIM + 1, tx + 1] = u_in[i + 1, j]
    if tx == 0 and i <= r_end + 1:
        smem[ty + 1, 0] = u_in[i, j - 1]
    if tx == TILE_DIM - 1 and j + 1 <= c_end + 1 and i <= r_end + 1:
        smem[ty + 1, TILE_DIM + 1] = u_in[i, j + 1]

    cuda.syncthreads()   # all warps must finish filling the tile before any stencil read

    if i >= r_start and i <= r_end and j >= c_start and j <= c_end:
        c = smem[ty + 1, tx + 1]
        up = smem[ty, tx + 1]
        dn = smem[ty + 2, tx + 1]
        lf = smem[ty + 1, tx]
        rt = smem[ty + 1, tx + 2]
        kc = kappa[i, j]
        k_dn = 0.5 * (kappa[i + 1, j] + kc)
        k_up = 0.5 * (kappa[i - 1, j] + kc)
        k_rt = 0.5 * (kappa[i, j + 1] + kc)
        k_lf = 0.5 * (kappa[i, j - 1] + kc)
        rhs = ((k_dn * (dn - c) - k_up * (c - up)) * inv_dy2
               + (k_rt * (rt - c) - k_lf * (c - lf)) * inv_dx2
               - lam * c * c * c + S[i, j])
        u_out[i, j] = c + dt * rhs


def launch_stencil(stream, u_in, u_out, st, r0, r1, c0, c1, p):
    nbx = (c1 - c0 + TILE_DIM) // TILE_DIM
    nby = (r1 - r0 + TILE_DIM) // TILE_DIM
    tiled_stencil_kernel[(nbx, nby), (TILE_DIM, TILE_DIM), stream](
        u_in, u_out, st["kappa"], st["S"], r0, r1, c0, c1, p["dt"], p["inv_dy2"], p["inv_dx2"], p["lam"])


def _swap(st):
    st["cur"], st["nxt"] = st["nxt"], st["cur"]


def step_serialized(ctx, st, p):
    """Baseline: everything on the default stream, no compute/communication overlap."""
    d_u, d_o = st[st["cur"]], st[st["nxt"]]
    reqs = exchange_halos_async(ctx, st, d_u, 0)
    finish_halo_exchange(ctx, st, d_u, 0, reqs)
    launch_stencil(0, d_u, d_o, st, 1, NY_LOC, 1, NX_LOC, p)
    cuda.synchronize()
    _swap(st)


def step_overlapped(ctx, st, p):
    """Dual-stream pipeline: interior on stream_interior overlaps MPI + H2D + boundary ring."""
    d_u, d_o = st[st["cur"]], st[st["nxt"]]
    sb, si = st["stream_boundary"], st["stream_interior"]
    ny, nx = NY_LOC, NX_LOC
    # Stage A + post MPI
    reqs = exchange_halos_async(ctx, st, d_u, sb)
    # Stage B: deep interior [2..ny-1, 2..nx-1] while messages are in flight
    launch_stencil(si, d_u, d_o, st, 2, ny - 1, 2, nx - 1, p)
    # Stage C: wait, H2D, unpack, then the four outer perimeter rings
    finish_halo_exchange(ctx, st, d_u, sb, reqs)
    launch_stencil(sb, d_u, d_o, st, 1, 1, 1, nx, p)             # row 1
    launch_stencil(sb, d_u, d_o, st, ny, ny, 1, nx, p)           # row ny
    launch_stencil(sb, d_u, d_o, st, 2, ny - 1, 1, 1, p)         # column 1
    launch_stencil(sb, d_u, d_o, st, 2, ny - 1, nx, nx, p)       # column nx
    sb.synchronize()
    si.synchronize()
    _swap(st)


# TASK 5: Reductions, PTX audit, benchmark, manifest
def build_device_state(host):
    nbuf = 2 * NX_LOC + 2 * NY_LOC
    st = {
        "u_a": cuda.to_device(host["u"]),
        "u_b": cuda.to_device(host["u"]),
        "scratch": cuda.to_device(host["u"]),
        "kappa": cuda.to_device(host["kappa"]),
        "S": cuda.to_device(host["S"]),
        "d_send": cuda.device_array(nbuf, dtype=np.float64),
        "d_recv": cuda.device_array(nbuf, dtype=np.float64),
        "h_send": cuda.pinned_array(nbuf, dtype=np.float64),
        "h_recv": cuda.pinned_array(nbuf, dtype=np.float64),
        "stream_boundary": cuda.stream(),
        "stream_interior": cuda.stream(),
        "cur": "u_a", "nxt": "u_b",
    }
    st["h_send"][:] = 0.0
    st["h_recv"][:] = 0.0
    return st


def reset_state(st, host):
    st["u_a"].copy_to_device(host["u"])
    st["u_b"].copy_to_device(host["u"])
    st["h_send"][:] = 0.0
    st["h_recv"][:] = 0.0   # PROC_NULL neighbors never write here -> ghost stays 0 (Dirichlet)
    st["cur"], st["nxt"] = "u_a", "u_b"
    cuda.synchronize()


def global_reduce(cart, value, op):
    send = np.array([value], dtype=np.float64)
    recv = np.zeros(1, dtype=np.float64)
    cart.Allreduce(send, recv, op=op)
    return float(recv[0])


def global_energy(ctx, st, host):
    u = st[st["cur"]].copy_to_host()
    e_loc = float(u[1:-1, 1:-1].sum()) * host["dx"] * host["dy"]
    return global_reduce(ctx["cart"], e_loc, MPI.SUM)


def global_residual(ctx, st, host, p):
    """L2 norm of f(u^n) = (stencil(u^n) - u^n)/dt (the discrete time derivative); non-destructive."""
    d_u = st[st["cur"]]
    reqs = exchange_halos_async(ctx, st, d_u, 0)
    finish_halo_exchange(ctx, st, d_u, 0, reqs)
    launch_stencil(0, d_u, st["scratch"], st, 1, NY_LOC, 1, NX_LOC, p)
    cuda.synchronize()
    u = d_u.copy_to_host()
    un = st["scratch"].copy_to_host()
    f = (un[1:-1, 1:-1] - u[1:-1, 1:-1]) / p["dt"]
    r_loc = float((f * f).sum()) * host["dx"] * host["dy"]
    return math.sqrt(global_reduce(ctx["cart"], r_loc, MPI.SUM))


def run_mode(ctx, host, st, p, mode):
    cart = ctx["cart"]
    step_fn = step_serialized if mode == "serialized" else step_overlapped

    # warm-up (JIT compile) then reset so both modes start from the identical state
    for _ in range(2):
        step_fn(ctx, st, p)
    reset_state(st, host)

    ev0, ev1 = cuda.event(timing=True), cuda.event(timing=True)
    energy, residual = {}, {}
    energy[0] = global_energy(ctx, st, host)
    residual[0] = global_residual(ctx, st, host, p)
    wall_ms, event_ms = 0.0, 0.0
    for seg in range(STEPS // REPORT_EVERY):
        cuda.synchronize()
        cart.Barrier()
        t0 = time.perf_counter()
        ev0.record()
        for _ in range(REPORT_EVERY):
            step_fn(ctx, st, p)
        ev1.record()
        ev1.synchronize()
        t1 = time.perf_counter()
        wall_ms += (t1 - t0) * 1e3
        event_ms += cuda.event_elapsed_time(ev0, ev1)
        step = (seg + 1) * REPORT_EVERY
        energy[step] = global_energy(ctx, st, host)
        residual[step] = global_residual(ctx, st, host, p)
    final_u = st[st["cur"]].copy_to_host()
    return {"energy": energy, "residual": residual, "wall_ms": wall_ms,
            "event_ms": event_ms, "final_u": final_u}


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    text = asm if isinstance(asm, str) else "\n".join(asm.values())
    counts = {
        "st.shared.f64": text.count("st.shared.f64"),
        "ld.shared.f64": text.count("ld.shared.f64"),
        "bar.sync": text.count("bar.sync"),
    }
    counts["ptx_audit_passed"] = all(counts[k] > 0 for k in ("st.shared.f64", "ld.shared.f64", "bar.sync"))
    return counts


def run_solver(ctx, host):
    cart = ctx["cart"]
    rank = ctx["rank"]
    dx = host["dx"]

    kmax = global_reduce(cart, float(host["kappa"][1:-1, 1:-1].max()), MPI.MAX)
    kmin = global_reduce(cart, -float(host["kappa"][1:-1, 1:-1].min()), MPI.MAX) * -1.0
    dt_max = dx * dx / (4.0 * kmax)
    dt = DT_SAFETY * dt_max
    p = {"dt": float(dt), "inv_dy2": 1.0 / (host["dy"] ** 2), "inv_dx2": 1.0 / (dx ** 2),
         "lam": float(LAMBDA_DAMP)}

    st = build_device_state(host)
    res_ser = run_mode(ctx, host, st, p, "serialized")
    reset_state(st, host)
    res_ovl = run_mode(ctx, host, st, p, "overlapped")

    diff_loc = float(np.max(np.abs(res_ser["final_u"][1:-1, 1:-1] - res_ovl["final_u"][1:-1, 1:-1])))
    parity = global_reduce(cart, diff_loc, MPI.MAX)

    t_ser_ev = global_reduce(cart, res_ser["event_ms"], MPI.MAX)
    t_ovl_ev = global_reduce(cart, res_ovl["event_ms"], MPI.MAX)
    t_ser_w = global_reduce(cart, res_ser["wall_ms"], MPI.MAX)
    t_ovl_w = global_reduce(cart, res_ovl["wall_ms"], MPI.MAX)

    ptx = ptx_audit()
    ptx_flags = [ptx["ptx_audit_passed"]]
    all_ptx = bool(global_reduce(cart, 1.0 if all(ptx_flags) else 0.0, MPI.MIN))

    if rank == 0:
        steps_list = sorted(res_ovl["energy"].keys())
        e_ovl = [res_ovl["energy"][s] for s in steps_list]
        monotone = all(e_ovl[k + 1] < e_ovl[k] for k in range(len(e_ovl) - 1))
        print("=" * 72)
        print("DISTRIBUTED MPI + CUDA STENCIL  |  4 ranks, 2x2 Cartesian, 512x512 global")
        print(f"GITHUB_USERNAME      : {ctx['github_user']}")
        print(f"GPU UUID             : {ctx['gpu_uuid']}")
        print(f"INSTANCE_SEED        : {ctx['seed']}")
        print(f"local padded shape   : ({PAD_Y}, {PAD_X})")
        print(f"dx = dy              : {dx:.10e}")
        print(f"kappa range (global) : [{kmin:.6f}, {kmax:.6f}]")
        print(f"dt_max = dx^2/(4kmax): {dt_max:.6e}")
        print(f"dt used              : {dt:.6e}   (dt/dt_max = {dt / dt_max:.3f})")
        print("-" * 72)
        print("Allreduce energy trace E = sum(u)*dx*dy and residual ||f(u)||_2")
        for s in steps_list:
            print(f"  step {s:4d} | E_serial = {res_ser['energy'][s]:.12e} | "
                  f"E_overlap = {res_ovl['energy'][s]:.12e} | residual = {res_ovl['residual'][s]:.6e}")
        print(f"energy strictly decreasing: {monotone}")
        print("-" * 72)
        print(f"parity_max_diff (serialized vs overlapped): {parity:.3e}  (< 1e-12: {parity < 1e-12})")
        print(f"serialized_ms (CUDA events, max over ranks): {t_ser_ev:.3f}   wall: {t_ser_w:.3f}")
        print(f"overlapped_ms (CUDA events, max over ranks): {t_ovl_ev:.3f}   wall: {t_ovl_w:.3f}")
        print(f"overlap speedup (events): {t_ser_ev / t_ovl_ev:.4f}x   (wall): {t_ser_w / t_ovl_w:.4f}x")
        print("-" * 72)
        print("PTX audit of tiled_stencil_kernel.inspect_asm():")
        print(f"  st.shared.f64 = {ptx['st.shared.f64']}")
        print(f"  ld.shared.f64 = {ptx['ld.shared.f64']}")
        print(f"  bar.sync      = {ptx['bar.sync']}")
        print(f"  ptx_audit_passed = {all_ptx}")
        print("=" * 72)

        manifest = {
            "GITHUB_USERNAME": ctx["github_user"],
            "github_username": ctx["github_user"],
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": str(ctx["seed"]),
            "mpi_ranks": 4,
            "process_grid": PROC_DIMS,
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [PAD_Y, PAD_X],
            "steps": STEPS,
            "lambda": LAMBDA_DAMP,
            "dx": dx,
            "kappa_max": kmax,
            "dt_max": dt_max,
            "dt": dt,
            "energy_trace_overlapped": {str(s): res_ovl["energy"][s] for s in steps_list},
            "energy_trace_serialized": {str(s): res_ser["energy"][s] for s in steps_list},
            "residual_trace_overlapped": {str(s): res_ovl["residual"][s] for s in steps_list},
            "energy_monotone_decreasing": monotone,
            "parity_max_diff": parity,
            "serialized_ms": t_ser_ev,
            "overlapped_ms": t_ovl_ev,
            "serialized_wall_ms": t_ser_w,
            "overlapped_wall_ms": t_ovl_w,
            "overlap_speedup": t_ser_ev / t_ovl_ev,
            "ptx_counts": {"st.shared.f64": ptx["st.shared.f64"],
                           "ld.shared.f64": ptx["ld.shared.f64"],
                           "bar.sync": ptx["bar.sync"]},
            "ptx_audit_passed": all_ptx,
        }
        with open("submission_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
        print("submission_manifest.json written.")
    cart.Barrier()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: distributed_cuda_solver.py <github_username>")
    github_user = sys.argv[1]
    ctx = init_distributed_context(github_user)
    host = initialize_subdomains(ctx)
    run_solver(ctx, host)


if __name__ == "__main__":
    main()
