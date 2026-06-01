import numpy as np

def verify_polyphase(T=2000, win=16, hop=4, seed=0, dtype=np.float32):
    assert win % hop == 0
    K = win // hop
    L = (T - 1) * hop + win

    rng = np.random.default_rng(seed)
    frame = rng.standard_normal((T, win)).astype(dtype)  # updates: [T, win]

    # ----------------------------
    # Baseline: overlap-add (scatter_add)
    # y[t*hop + n] += frame[t, n]
    # ----------------------------
    t = np.arange(T, dtype=np.int64)[:, None]      # [T,1]
    n = np.arange(win, dtype=np.int64)[None, :]    # [1,win]
    idx = (t * hop + n).reshape(-1)                # [T*win]
    upd = frame.reshape(-1)

    y_ref = np.zeros((L,), dtype=dtype)
    np.add.at(y_ref, idx, upd)                     # correct with duplicates

    # ----------------------------
    # Polyphase: K scatters (no add) + sum
    # n = q*hop + r, r in [0..hop-1]
    # idx_q(t,r) = (t+q)*hop + r   (no duplicates within fixed q)
    # ----------------------------
    frame_q = frame.reshape(T, K, hop)             # [T, K, hop]
    r = np.arange(hop, dtype=np.int64)[None, :]    # [1,hop]

    y_poly = np.zeros((L,), dtype=dtype)
    for q in range(K):
        idx_q = (((np.arange(T, dtype=np.int64)[:, None] + q) * hop) + r).reshape(-1)  # [T*hop]
        upd_q = frame_q[:, q, :].reshape(-1)

        # "non-add scatter": assignment is safe because idx_q has no duplicates
        yq = np.zeros((L,), dtype=dtype)
        yq[idx_q] = upd_q
        y_poly += yq

        # Optional: assert uniqueness (cheap-ish for moderate T)
        # assert np.unique(idx_q).size == idx_q.size

    # ----------------------------
    # Compare
    # ----------------------------
    diff = np.abs(y_ref - y_poly)
    print(f"T={T}, win={win}, hop={hop}, K={K}, L={L}")
    print("max_abs_err:", diff.max())
    print("mean_abs_err:", diff.mean())
    print("exact_equal:", np.array_equal(y_ref, y_poly))

    # Show a small overlap-heavy slice for sanity
    pos = 32
    print("y_ref slice :", y_ref[pos-10:pos+20])
    print("y_poly slice:", y_poly[pos-10:pos+20])

    return y_ref, y_poly

if __name__ == "__main__":
    verify_polyphase(T=122881, win=16, hop=4, seed=0)