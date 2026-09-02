import numpy as np

from copy import deepcopy

# sigmoid function
sig = lambda z: 1.0 / (1.0 + np.exp(-z))


class LSTM:
    """
    a single-layer lstm trained by truncated bptt

    states are column vectors (with h, and c having shape (N_units, 1))
    gate parameters carry a suffix in {f i g o}, which is forget/input/candidate/output
    all gates but g use sigmoid, while g uses tanh

    sequences are time-major, X has shape (T, N_dim_in)
    """

    # training defaults, overridable from __init__ or train kwargs
    N_wash = 100  # washout steps before bptt starts
    seq_len = 200  # truncation length
    epochs = 40
    lr = 3e-3
    clip = 5.0  # global grad-norm clip
    val_frac = 0.1  # tail of each segment held out for closed-loop validation
    forget_bias = 1.0
    norm_method = "range"
    recurrent_init = "orthogonal"

    def __init__(self, N_dim_in, N_units, seed=0, **kwargs):
        self.N_dim_in = N_dim_in  # observable/physical dim
        self.N_units = N_units  # the "reservoir" size (high dimensional latent size)

        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(LSTM, key)]

        self.seed = seed
        rng = self.rng = np.random.default_rng(seed)

        # uniform(-k, k) is seemingly standard for tanh/sigmoid
        k = 1.0 / np.sqrt(N_units)
        rand = lambda r, c: rng.uniform(-k, k, size=(r, c))

        # the gates can be put in a dictionary for easier access
        self.p = {}
        for gate in "figo":
            # input weight matrix
            self.p["W" + gate] = rand(N_units, N_dim_in)
            # recurrent weight matrix
            self.p["U" + gate] = self._recurrent_matrix(rng)
            # bias vector
            self.p["b" + gate] = np.zeros((N_units, 1))

        # the forget gate should default to retaining memory
        self.p["bf"] += self.forget_bias

        # the readout matrix and bias
        self.p["Wy"] = rand(N_dim_in, N_units)
        self.p["by"] = np.zeros((N_dim_in, 1))

        self.history = {"train": [], "val": []}

    def _recurrent_matrix(self, rng):
        # apparently an orthogonal U keeps the recurrent jacobian near unit gain over long prediction periods,
        # kinda like why esn rescales W by spectral
        if self.recurrent_init == "orthogonal":
            Q, R = np.linalg.qr(rng.normal(size=(self.N_units, self.N_units)))
            return Q * np.sign(np.diag(R))
        elif self.recurrent_init == "uniform":
            k = 1.0 / np.sqrt(self.N_units)
            return rng.uniform(-k, k, size=(self.N_units, self.N_units))
        raise ValueError(
            f"recurrent_init {self.recurrent_init} not implemented ['orthogonal', 'uniform']"
        )

    def copy(self):
        return deepcopy(self)

    @property
    def trained(self):
        """Flag to check if the model has been trained"""
        return getattr(self, "_trained", False)

    @property
    def norm(self) -> np.ndarray:
        if not hasattr(self, "_norm"):
            return np.ones((self.N_dim_in, 1))
        return self._norm

    @norm.setter
    def norm(self, value):
        value = np.asarray(value, dtype=float).reshape(self.N_dim_in, 1).copy()
        value[abs(value) < 1e-12] = 1.0  # prevent division by zero
        self._norm = value

    @property
    def shift(self) -> np.ndarray:
        if not hasattr(self, "_shift"):
            return np.zeros((self.N_dim_in, 1))
        return self._shift

    @shift.setter
    def shift(self, value):
        self._shift = np.asarray(value, dtype=float).reshape(self.N_dim_in, 1).copy()

    def set_norm(self, data):
        self.norm, self.shift = self._set_norm(
            self._as_sequences(data), method=self.norm_method
        )

    @staticmethod
    def _set_norm(sequences, method="range"):
        """
        Per-component shift and scale over all training segments.

        Args:
            sequences (list): list of (T, N_dim_in, 1) arrays.
            method (str): one of [None, 'std', 'max', 'mean', 'range'].
        Returns:
            tuple: (norm, shift), both of shape (N_dim_in, 1).
        """
        U = np.concatenate(sequences, axis=0)

        if method is None:
            return np.ones(U.shape[1:]), np.zeros(U.shape[1:])

        shift = U.mean(axis=0)
        Us = U - shift

        if method == "std":
            norm = Us.std(axis=0)
        elif method == "max":
            norm = Us.max(axis=0)
        elif method == "mean":
            norm = abs(Us).mean(axis=0)
        elif method == "range":
            norm = Us.max(axis=0) - Us.min(axis=0)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return norm, shift

    def normalize_input(self, data):
        return (data - self.shift) / self.norm

    def denormalize_output(self, data):
        return data * self.norm + self.shift

    def _as_sequences(self, data) -> list:
        """
        Accepts (T, N_dim_in), (T, N_dim_in, 1), (L, T, N_dim_in) or (L, T, N_dim_in, 1)
        and returns a list of L arrays of shape (T, N_dim_in, 1).
        """
        if isinstance(data, list):
            return [s for d in data for s in self._as_sequences(d)]

        data = np.asarray(data, dtype=float)

        if data.ndim == 2:
            data = data[np.newaxis, ..., np.newaxis]
        elif data.ndim == 3 and data.shape[1] == self.N_dim_in and data.shape[-1] == 1:
            data = data[np.newaxis]
        elif data.ndim == 3:
            data = data[..., np.newaxis]
        elif data.ndim != 4:
            raise ValueError(
                f"data has shape {data.shape}, expected 2 to 4 dimensions "
                f"[(L) x T x {self.N_dim_in} x (1)]"
            )

        assert data.shape[2:] == (
            self.N_dim_in,
            1,
        ), f"data has shape {data.shape}, expected trailing dims ({self.N_dim_in}, 1)"

        return list(data)

    # only a single step, returns updated (h, c) and the readout
    # h and c are technically h_{t-1} and c_{t-1}
    def step(self, x: np.ndarray, h: np.ndarray, c: np.ndarray):
        if x.ndim == 1:
            x = x[:, np.newaxis]

        assert x.shape == (
            self.N_dim_in,
            1,
        ), f"x has incorrect shape {x.shape}, expected {(self.N_dim_in, 1)}"

        assert h.shape == (
            self.N_units,
            1,
        ), f"h has incorrect shape {h.shape}, expected {(self.N_units, 1)}"
        assert c.shape == (
            self.N_units,
            1,
        ), f"c has incorrect shape {c.shape}, expected {(self.N_units, 1)}"

        p = self.p
        # all activations follow a = W*x + U*h + b
        af = p["Wf"] @ x + p["Uf"] @ h + p["bf"]
        ai = p["Wi"] @ x + p["Ui"] @ h + p["bi"]
        ag = p["Wg"] @ x + p["Ug"] @ h + p["bg"]
        ao = p["Wo"] @ x + p["Uo"] @ h + p["bo"]
        f, i, g, o = sig(af), sig(ai), np.tanh(ag), sig(ao)

        c_new = f * c + i * g

        # for caching purposes
        tc = np.tanh(c_new)

        h_new = o * tc
        # the final readout
        y = p["Wy"] @ h_new + p["by"]

        cache = (x, h, c, c_new, tc, f, i, g, o)  # backprop needs this cached
        return h_new, c_new, y, cache

    # goes forward over a whole input sequence, passing in the real x
    # X is (T, N_dim_in, 1)
    def forward(self, X: np.ndarray, h0=None, c0=None):

        assert X.shape[1:] == (
            self.N_dim_in,
            1,
        ), f"X has shape {X.shape}, expected (T, {self.N_dim_in}, 1)"

        h = np.zeros((self.N_units, 1)) if h0 is None else h0
        c = np.zeros((self.N_units, 1)) if c0 is None else c0

        assert h.shape == (self.N_units, 1) and c.shape == (
            self.N_units,
            1,
        ), f"h or c with shape {h.shape} and {c.shape}. need {(self.N_units, 1)}"

        # length of the input sequence
        T = X.shape[0]

        Y, caches = np.empty((T, self.N_dim_in, 1)), []
        for t in range(T):
            h, c, Y[t], cache = self.step(X[t], h, c)
            caches.append(cache)
        return Y, caches, (h, c)

    def backward(self, caches: list, dY: np.ndarray):
        p = self.p

        # gradients of all the mats and vecs in p
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        # for the "temporal" part of the backprop
        dh_next = np.zeros((self.N_units, 1))
        dc_next = np.zeros((self.N_units, 1))

        for t in reversed(range(len(caches))):
            x, h_prev, c_prev, c, tc, f, i, g, o = caches[t]
            h_t = o * tc
            grads["Wy"] += dY[t] @ h_t.T  # h_t = o*tc
            grads["by"] += dY[t]

            dh = p["Wy"].T @ dY[t] + dh_next
            do = dh * tc
            dc = dh * o * (1 - tc**2) + dc_next

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_next = dc * f

            daf = df * f * (1 - f)
            dai = di * i * (1 - i)
            dao = do * o * (1 - o)
            dag = dg * (1 - g**2)

            for gate, da in zip("figo", (daf, dai, dag, dao)):
                grads["W" + gate] += da @ x.T  # (3.4) accumulate
                grads["U" + gate] += da @ h_prev.T
                grads["b" + gate] += da
            dh_next = (
                p["Uf"].T @ daf + p["Ui"].T @ dai + p["Ug"].T @ dag + p["Uo"].T @ dao
            )
        return grads

    def _rollout(self, x: np.ndarray, Nt: int, state):
        """closed loop in normalized space, each readout is fed back in as the next input"""
        h, c = state
        out = np.empty((Nt, self.N_dim_in, 1))
        for t in range(Nt):
            h, c, x, _ = self.step(x, h, c)
            out[t] = x
        return out, (h, c)

    def train(self, data, verbose=True, **kwargs):
        """
        Truncated BPTT with Adam, one-step-ahead loss, model selection on closed-loop
        validation error.

        Args:
            data (np.ndarray): time series with dimensions [(L) x T x N_dim_in x (1)].
            verbose (bool): print per-epoch losses.
            **kwargs: overrides any class attribute (epochs, lr, seq_len, N_wash, ...).
        """
        for key, val in kwargs.items():
            if hasattr(self, key):
                print(f"modyfing {key} = {getattr(self, key)} -> {val} at training")
                setattr(self, key, val)

        sequences = self._as_sequences(data)

        # hold out the end of each segment, its washout comes from the training part
        N_val = [int(round(self.val_frac * len(U))) for U in sequences]
        train_raw = [U[: len(U) - n] for U, n in zip(sequences, N_val)]
        val_raw = [
            U[len(U) - n - self.N_wash :]
            for U, n in zip(sequences, N_val)
            if n > 0 and len(U) - n >= self.N_wash
        ]

        for ll, U in enumerate(train_raw):
            if len(U) < self.N_wash + self.seq_len + 1:
                raise ValueError(
                    f"Segment {ll} is too short for training: {len(U)} < "
                    f"N_wash + seq_len + 1 = {self.N_wash + self.seq_len + 1}"
                )

        # norm is computed on the training part only
        self.norm, self.shift = self._set_norm(train_raw, method=self.norm_method)

        U_train = [self.normalize_input(U) for U in train_raw]
        U_val = [self.normalize_input(U) for U in val_raw]

        adam = {
            k: [np.zeros_like(v), np.zeros_like(v)]  # the m and v moments
            for k, v in self.p.items()
        }
        tstep = 0
        best_loss, best_p = np.inf, None

        for ep in range(self.epochs):
            total, N_batch = 0.0, 0

            for U in U_train:
                h = np.zeros((self.N_units, 1))
                c = np.zeros((self.N_units, 1))

                # the washout to warm it up, like the esn
                for t in range(self.N_wash):
                    h, c, _, _ = self.step(U[t], h, c)

                ptr = self.N_wash
                while ptr + self.seq_len + 1 <= len(U):

                    X = U[ptr : ptr + self.seq_len]  # inputs  x_t
                    Yt = U[ptr + 1 : ptr + self.seq_len + 1]  # targets x_{t+1}

                    Yh, caches, (h, c) = self.forward(X, h, c)

                    r = Yh - Yt
                    total += 0.5 * np.mean(r**2)
                    N_batch += 1

                    tstep += 1
                    grads = self.backward(caches, r / (r.shape[0] * self.N_dim_in))
                    self._adam_step(grads, adam, tstep)

                    # carries the state across the truncation boundary but not the graph
                    h, c = h.copy(), c.copy()

                    ptr += self.seq_len

            train_loss = total / max(N_batch, 1)
            val_loss = self._validate(U_val) if U_val else train_loss

            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)

            if val_loss < best_loss:
                best_loss, best_p = val_loss, deepcopy(self.p)

            if verbose:
                print(f"epoch {ep:3d}  loss {train_loss:.4e}  val nRMSE {val_loss:.4e}")

        # keep the best epoch rather than the last one
        if best_p is not None:
            self.p = best_p

        self._trained = True

    def _adam_step(self, grads, adam, tstep, b1=0.9, b2=0.999, eps=1e-8):
        gnorm = np.sqrt(sum((gr**2).sum() for gr in grads.values()))

        # global norm clip is apparently good
        scale = self.clip / max(gnorm, self.clip)

        for key, gr in grads.items():
            gr = gr * scale
            m_, v_ = adam[key]
            m_[:] = b1 * m_ + (1 - b1) * gr
            v_[:] = b2 * v_ + (1 - b2) * gr * gr
            mhat = m_ / (1 - b1**tstep)
            vhat = v_ / (1 - b2**tstep)
            self.p[key] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def _validate(self, sequences):
        scores = []
        for U in sequences:
            Y_wash, _, state = self.forward(U[: self.N_wash])
            Y_closed, _ = self._rollout(Y_wash[-1], len(U) - self.N_wash - 1, state)
            Y = np.concatenate((Y_wash[-1:], Y_closed), axis=0)

            score = self.compute_nRMSE(U[self.N_wash :], Y)
            # prevents a diverging run from ruining the mean
            scores.append(score if np.isfinite(score) else 1e10)

        return float(np.mean(scores))

    def compute_nRMSE(self, Y_true, Y_pred, norm=1.0):
        # same definition as EchoStateNetwork.compute_nRMSE, so the two are comparable
        return np.mean(np.sqrt((Y_true - Y_pred) ** 2)) / np.mean(np.sqrt(norm**2))

    def gradient_check(self, data, n_probe=6, eps=1e-5, seed=0):
        """
        Central-difference check of backward() against forward(). Returns the worst
        relative error over n_probe random entries of each parameter
        """
        sequences = self._as_sequences(data)
        if self.trained:
            norm, shift = self.norm, self.shift
        else:
            norm, shift = self._set_norm(sequences, method=self.norm_method)

        U = (sequences[0] - shift) / norm
        X, Yt = U[:-1], U[1:]

        def loss():
            Yh, caches, _ = self.forward(X)
            return 0.5 * np.mean((Yh - Yt) ** 2), Yh, caches

        L, Yh, caches = loss()
        grads = self.backward(caches, (Yh - Yt) / (Yh.shape[0] * self.N_dim_in))
        floor = 100.0 * np.finfo(float).eps * max(L, 1.0) / eps

        rng = np.random.default_rng(seed)
        worst = 0.0

        for key, P in self.p.items():
            for _ in range(n_probe):
                idx = tuple(rng.integers(0, s) for s in P.shape)
                p0 = P[idx]

                P[idx] = p0 + eps
                Lp = loss()[0]
                P[idx] = p0 - eps
                Lm = loss()[0]
                P[idx] = p0

                num, ana = (Lp - Lm) / (2 * eps), grads[key][idx]
                if abs(num) + abs(ana) < floor:
                    continue
                worst = max(worst, abs(num - ana) / (abs(num) + abs(ana)))

        return worst

    # warms up on a true sequence and then returns the final (h, c) and prediction
    def openLoop(self, X_phys):
        X = self.normalize_input(self._as_sequences(X_phys)[0])
        Y, _, state = self.forward(X)
        return self.denormalize_output(Y), state

    def closedLoop(self, x0_phys, Nt, state):
        x = np.asarray(x0_phys, dtype=float).reshape(self.N_dim_in, 1)
        out, state = self._rollout(self.normalize_input(x), Nt, state)
        return self.denormalize_output(out), state
