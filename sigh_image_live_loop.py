import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.cm as cm

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32


# -----------------------------------------------------------------------------
# Graphical equalizer
# -----------------------------------------------------------------------------

class GraphicalEQ(tk.Frame):
    """Simple draggable 10-band radial-frequency gain curve."""

    def __init__(self, parent, num_bands=10, width=600, height=90, callback=None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.num_bands = num_bands
        self.callback = callback

        self.canvas = tk.Canvas(
            self,
            width=self.width,
            height=self.height,
            bg="#2E2E2E",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.gains = np.array(
            [0.1, 0.4, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.4, 0.1],
            dtype=np.float32,
        )
        if num_bands != 10:
            self.gains = np.ones(num_bands, dtype=np.float32)

        self.band_width = self.width / self.num_bands
        self.selected_band = None

        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.draw()

    def _on_click(self, event):
        band_index = int(event.x // self.band_width)
        if 0 <= band_index < self.num_bands:
            self.selected_band = band_index
            self._update_gain(event.y)

    def _on_release(self, _event):
        self.selected_band = None

    def _on_drag(self, event):
        if self.selected_band is not None:
            self._update_gain(event.y)

    def _update_gain(self, y_pos):
        y_clamped = max(0, min(self.height, y_pos))
        gain = 1.0 - (y_clamped / self.height)
        self.gains[self.selected_band] = gain
        self.draw()
        if self.callback:
            self.callback()

    def draw(self):
        self.canvas.delete("all")
        points = self._get_curve_points_for_drawing()
        self.canvas.create_polygon(points, fill="#4A90E2", outline="")

        for i, gain in enumerate(self.gains):
            x = (i + 0.5) * self.band_width
            y = (1.0 - float(gain)) * self.height
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill="white", outline="black"
            )

    def _get_curve_points_for_drawing(self):
        curve_points = [0, self.height]
        x_coords = np.linspace(0, self.width, self.width)
        band_centers_x = (np.arange(self.num_bands) + 0.5) * self.band_width
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)

        for x, gain in zip(x_coords, interp_gains):
            y = (1.0 - gain) * self.height
            curve_points.extend([x, y])

        curve_points.extend([self.width, self.height])
        return curve_points

    def get_filter_shape_tensor(self, num_points=256):
        """
        Return the current EQ curve as a 1-D gain lookup table.

        NOTE:
        The original Sigh code indexes this curve by normalized k^2
        (squared radial spatial frequency), not by linear radius.
        This version preserves that behavior intentionally.
        """
        x_coords = np.linspace(0, 1, num_points)
        band_centers_x = np.linspace(0, 1, self.num_bands)
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        return torch.tensor(
            interp_gains,
            dtype=torch.float32,
            device=device
        )


# -----------------------------------------------------------------------------
# Frequency-domain field
# -----------------------------------------------------------------------------

class EnhancedHolographicField(nn.Module):
    def __init__(self, dimensions=(64, 64)):
        super().__init__()
        self.dimensions = dimensions

        k_freq = [
            torch.fft.fftfreq(n, d=1 / n, dtype=torch.float32)
            for n in dimensions
        ]
        k_grid = torch.meshgrid(*k_freq, indexing="ij")
        k2_tensor = sum(k ** 2 for k in k_grid)

        # Preserve the original Sigh convention:
        # normalized squared radial frequency, 0..1.
        self.register_buffer("k2", k2_tensor / k2_tensor.max())

    def evolve(self, field_state, steps=1, custom_filter_shape=None):
        with torch.no_grad():
            field_fft = torch.fft.fft2(field_state.float())

            if custom_filter_shape is not None:
                num_points = len(custom_filter_shape)
                indices = (
                    self.k2 * (num_points - 1)
                ).long().clamp(0, num_points - 1)
                final_filter = custom_filter_shape[indices]
            else:
                final_filter = torch.exp(-self.k2)

            for _ in range(steps):
                field_fft = field_fft * final_filter

            return torch.fft.ifft2(field_fft).real.to(torch_dtype)


# -----------------------------------------------------------------------------
# Live recursive app
# -----------------------------------------------------------------------------

class LiveCollapseApp:
    """
    Continuous recursive self-feed loop.

        x_(n+1) = alpha * x_original + (1-alpha) * F(x_n)

    The loop runs until Stop is pressed.

    Important:
    - The display is normalized for visibility.
    - Raw mean/std/min/max are measured BEFORE display normalization.
    - No "detected structure" metric is used.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Sigh Image — Live Recursive Loop")
        self.root.geometry("1250x900")

        self.field_dims = (64, 64)
        self.field = EnhancedHolographicField(self.field_dims).to(device)

        self.original_image = None
        self.original_tensor = None
        self.state = None

        self.running = False
        self.after_id = None
        self.generation = 0

        self.std_history = []
        self.entropy_history = []
        self.effective_modes_history = []

        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def setup_gui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(main, text="Controls", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))

        file_row = ttk.Frame(controls)
        file_row.pack(fill=tk.X, pady=4)

        ttk.Button(
            file_row, text="Load Image", command=self.load_image
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            file_row, text="Reset Loop", command=self.reset_loop
        ).pack(side=tk.LEFT, padx=5)

        self.status_text = tk.StringVar(
            value=f"Ready — device: {device}"
        )
        ttk.Label(
            file_row, textvariable=self.status_text
        ).pack(side=tk.LEFT, padx=18)

        # EQ
        eq_frame = ttk.LabelFrame(
            controls,
            text="Graphical Frequency Filter (low k² → high k²)",
            padding=5,
        )
        eq_frame.pack(fill=tk.X, pady=5)

        self.eq_widget = GraphicalEQ(
            eq_frame,
            width=680,
            height=90,
            num_bands=10,
            callback=self.on_filter_changed,
        )
        self.eq_widget.pack(padx=5, pady=5, anchor="center")

        preset_row = ttk.Frame(controls)
        preset_row.pack(fill=tk.X, pady=5)

        ttk.Button(
            preset_row, text="Low Pass", command=self.preset_low_pass
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            preset_row, text="High Pass", command=self.preset_high_pass
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            preset_row, text="Band Pass", command=self.preset_band_pass
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            preset_row, text="Notch", command=self.preset_notch
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            preset_row, text="All Pass", command=self.preset_all_pass
        ).pack(side=tk.LEFT, padx=5)

        # Live recursive loop
        loop_frame = ttk.LabelFrame(
            controls,
            text="Live Recursive Loop — runs until stopped",
            padding=10,
        )
        loop_frame.pack(fill=tk.X, pady=5)

        ttk.Label(
            loop_frame,
            text="Fresh-data α (0 = closed loop, 1 = original every step):",
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.alpha_var = tk.StringVar(value="0.0")
        ttk.Entry(
            loop_frame, textvariable=self.alpha_var, width=6
        ).pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(
            loop_frame, text="Step interval (ms):"
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.interval_var = tk.StringVar(value="100")
        ttk.Entry(
            loop_frame, textvariable=self.interval_var, width=6
        ).pack(side=tk.LEFT, padx=(0, 14))

        self.renorm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            loop_frame,
            text="Renormalize each generation",
            variable=self.renorm_var,
        ).pack(side=tk.LEFT, padx=(0, 14))

        self.start_button = ttk.Button(
            loop_frame, text="Start", command=self.start_loop
        )
        self.start_button.pack(side=tk.LEFT, padx=4)

        self.stop_button = ttk.Button(
            loop_frame, text="Stop", command=self.stop_loop
        )
        self.stop_button.pack(side=tk.LEFT, padx=4)

        # Displays
        display = ttk.Frame(main)
        display.pack(fill=tk.BOTH, expand=True)

        for r in range(2):
            display.grid_rowconfigure(r, weight=1)
        for c in range(2):
            display.grid_columnconfigure(c, weight=1)

        self.original_label = self._make_panel(
            display, "Original", 0, 0
        )
        self.current_label = self._make_panel(
            display, "Live recursive state", 0, 1
        )
        self.spectrum_label = self._make_panel(
            display, "Current log Fourier power", 1, 0
        )
        self.metrics_label = self._make_panel(
            display, "Live metrics", 1, 1
        )

    def _make_panel(self, parent, title, row, col):
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        frame.grid(
            row=row, column=col,
            sticky="nsew",
            padx=5, pady=5
        )
        label = ttk.Label(frame)
        label.pack(expand=True)
        return label

    # ------------------------------------------------------------------
    # Image setup
    # ------------------------------------------------------------------

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self.stop_loop()

            pil_image = Image.open(file_path).convert("RGB")
            self.original_image = np.array(pil_image)
            self.original_tensor = self._prepare_tensor(self.original_image)

            self._update_display(
                self.original_label,
                self.original_image,
                colormap=None,
            )

            self.reset_loop()

            self.status_text.set(
                f"Loaded {PathLikeName(file_path)} — device: {device}"
            )
        except Exception as exc:
            self.status_text.set(f"Load error: {exc}")

    def _prepare_tensor(self, image_array):
        if image_array.ndim == 3:
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array

        resized = cv2.resize(gray, self.field_dims)
        mn = float(resized.min())
        mx = float(resized.max())
        norm = (resized - mn) / (mx - mn + 1e-8)

        return torch.from_numpy(norm).float().to(device)

    # ------------------------------------------------------------------
    # Live loop controls
    # ------------------------------------------------------------------

    def start_loop(self):
        if self.original_tensor is None:
            self.status_text.set("Load an image first.")
            return

        if self.running:
            return

        if self.state is None:
            self.state = self.original_tensor.clone()

        self.running = True
        self.status_text.set(
            f"RUNNING — generation {self.generation}"
        )
        self._schedule_next_step(immediate=True)

    def stop_loop(self):
        self.running = False

        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.original_tensor is not None:
            self.status_text.set(
                f"STOPPED — generation {self.generation}"
            )

    def reset_loop(self):
        self.stop_loop()

        if self.original_tensor is None:
            return

        self.state = self.original_tensor.clone()
        self.generation = 0

        self.std_history.clear()
        self.entropy_history.clear()
        self.effective_modes_history.clear()

        self._render_state(self.state)
        self.status_text.set("Reset to original — generation 0")

    def _schedule_next_step(self, immediate=False):
        if not self.running:
            return

        try:
            interval = max(1, int(self.interval_var.get()))
        except ValueError:
            interval = 100
            self.interval_var.set("100")

        delay = 1 if immediate else interval
        self.after_id = self.root.after(delay, self._live_step)

    def _live_step(self):
        self.after_id = None

        if not self.running or self.state is None:
            return

        try:
            alpha = float(self.alpha_var.get())
            alpha = float(np.clip(alpha, 0.0, 1.0))
        except ValueError:
            alpha = 0.0
            self.alpha_var.set("0.0")

        try:
            filter_shape = self.eq_widget.get_filter_shape_tensor()

            filtered = self.field.evolve(
                self.state,
                steps=1,
                custom_filter_shape=filter_shape,
            )

            # This is the recursive inheritance rule.
            next_state = (
                alpha * self.original_tensor
                + (1.0 - alpha) * filtered
            )

            if self.renorm_var.get():
                mn = next_state.min()
                mx = next_state.max()
                next_state = (
                    next_state - mn
                ) / (mx - mn + 1e-8)

            self.state = next_state
            self.generation += 1

            self._render_state(self.state)

            self.status_text.set(
                f"RUNNING — generation {self.generation} — α={alpha:.3f}"
            )

        except Exception as exc:
            self.running = False
            self.status_text.set(f"Loop error: {exc}")
            return

        self._schedule_next_step()

    # ------------------------------------------------------------------
    # Measurements / display
    # ------------------------------------------------------------------

    def _render_state(self, tensor_state):
        arr = tensor_state.float().cpu().numpy()

        # Raw metrics: BEFORE display normalization.
        raw_mean = float(arr.mean())
        raw_std = float(arr.std())
        raw_min = float(arr.min())
        raw_max = float(arr.max())

        # Fourier power.
        fft = np.fft.fftshift(np.fft.fft2(arr))
        power = np.abs(fft) ** 2
        power_sum = float(power.sum()) + 1e-30

        p = power.ravel() / power_sum
        p_nonzero = p[p > 0]

        spectral_entropy = float(
            -np.sum(p_nonzero * np.log(p_nonzero + 1e-30))
        )

        # Participation ratio = effective number of powered modes.
        effective_modes = float(
            (power_sum ** 2)
            / (float(np.sum(power ** 2)) + 1e-30)
        )

        self.std_history.append(raw_std)
        self.entropy_history.append(spectral_entropy)
        self.effective_modes_history.append(effective_modes)

        # State display is normalized ONLY for visibility.
        self._update_display(
            self.current_label,
            arr,
            colormap="inferno",
        )

        # Log-power spectrum display.
        log_power = np.log1p(power)
        self._update_display(
            self.spectrum_label,
            log_power,
            colormap="magma",
        )

        metrics_img = self._metrics_image(
            generation=self.generation,
            raw_mean=raw_mean,
            raw_std=raw_std,
            raw_min=raw_min,
            raw_max=raw_max,
            entropy=spectral_entropy,
            effective_modes=effective_modes,
        )
        self._update_display(
            self.metrics_label,
            metrics_img,
            colormap=None,
        )

    def _metrics_image(
        self,
        generation,
        raw_mean,
        raw_std,
        raw_min,
        raw_max,
        entropy,
        effective_modes,
    ):
        img = np.zeros((330, 520, 3), dtype=np.uint8)

        lines = [
            "LIVE RECURSIVE LOOP",
            f"generation          {generation}",
            "",
            f"raw mean            {raw_mean:.8f}",
            f"raw std             {raw_std:.8f}",
            f"raw min             {raw_min:.8f}",
            f"raw max             {raw_max:.8f}",
            "",
            f"spectral entropy    {entropy:.6f}",
            f"effective modes     {effective_modes:.2f}",
            "",
            "Display contrast is normalized.",
            "Metrics above are not.",
        ]

        y = 34
        for i, line in enumerate(lines):
            scale = 0.72 if i == 0 else 0.58
            color = (255, 255, 255) if i == 0 else (205, 205, 205)
            cv2.putText(
                img,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 24

        return img

    def _update_display(self, label, data, colormap=None, size=(500, 320)):
        img = self.numpy_to_tkimage(
            data,
            size=size,
            colormap=colormap,
        )
        label.config(image=img)
        label.image = img

    def numpy_to_tkimage(self, array, size=(500, 320), colormap=None):
        if array.ndim == 3 and array.shape[2] == 3:
            arr = np.clip(array, 0, 255).astype(np.uint8)
        elif array.ndim == 2:
            mn = float(np.min(array))
            mx = float(np.max(array))
            norm = (array - mn) / (mx - mn + 1e-12)

            if colormap is None:
                arr = (norm * 255).astype(np.uint8)
            else:
                mapped = getattr(cm, colormap)(norm)[:, :, :3]
                arr = (mapped * 255).astype(np.uint8)
        else:
            arr = np.clip(array * 255, 0, 255).astype(np.uint8)

        pil = Image.fromarray(arr).resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    # ------------------------------------------------------------------
    # EQ presets
    # ------------------------------------------------------------------

    def on_filter_changed(self):
        # If stopped, redraw the CURRENT recursive state through no extra step;
        # the new filter takes effect on the next live generation.
        if self.state is not None and not self.running:
            self._render_state(self.state)

    def _set_gains(self, gains):
        self.eq_widget.gains = np.array(gains, dtype=np.float32)
        self.eq_widget.draw()
        self.on_filter_changed()

    def preset_low_pass(self):
        self._set_gains(
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        )

    def preset_high_pass(self):
        self._set_gains(
            [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        )

    def preset_band_pass(self):
        self._set_gains(
            [0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 0.8, 0.4, 0.2, 0.1]
        )

    def preset_notch(self):
        self._set_gains(
            [1.0, 1.0, 1.0, 0.2, 0.05, 0.05, 0.2, 1.0, 1.0, 1.0]
        )

    def preset_all_pass(self):
        self._set_gains(np.ones(10, dtype=np.float32))

    def on_close(self):
        self.stop_loop()
        self.root.destroy()


def PathLikeName(path):
    """Windows- and POSIX-friendly basename without importing pathlib."""
    return str(path).replace("\\", "/").split("/")[-1]


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveCollapseApp(root)
    root.mainloop()
