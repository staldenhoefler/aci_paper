from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_tour_gif(env, tour, filepath, title="Best Tour", fps=4):
    """
    Animate step-by-step construction of *tour* and save as GIF.
    Each frame adds one edge; the current city is highlighted.
    Returns the resolved Path.
    """
    coords = env.coords          # (n, 2) float32 in [0, 1]
    n_steps = len(tour) - 1     # number of edges
    show_labels = env.n <= 25

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    fig.patch.set_facecolor("white")

    def draw_frame(frame):
        ax.clear()
        ax.set_xlim(-0.07, 1.07)
        ax.set_ylim(-0.07, 1.07)
        ax.set_aspect("equal")
        ax.axis("off")

        # All cities
        ax.scatter(coords[:, 0], coords[:, 1],
                   c="#2196F3", s=70, zorder=5, edgecolors="white", linewidths=0.5)
        # Depot
        ax.scatter(coords[0, 0], coords[0, 1],
                   c="#F44336", s=220, marker="*", zorder=6)

        # Optional city labels
        if show_labels:
            for i, (x, y) in enumerate(coords):
                ax.text(x, y + 0.03, str(i), ha="center", va="bottom",
                        fontsize=7, color="#333333", zorder=8)

        # Edges constructed so far
        for i in range(frame):
            c1, c2 = tour[i], tour[i + 1]
            ax.plot([coords[c1, 0], coords[c2, 0]],
                    [coords[c1, 1], coords[c2, 1]],
                    "-", color="#4CAF50", linewidth=1.8, alpha=0.85, zorder=3)

        # Current city highlight
        if frame < len(tour):
            curr = tour[frame]
            ax.scatter(coords[curr, 0], coords[curr, 1],
                       c="#FFC107", s=180, zorder=7,
                       edgecolors="#333333", linewidths=1.2)

        # Running cost in subtitle
        running_cost = sum(
            float(env.get_cost_matrix(t)[tour[t], tour[t + 1]])
            for t in range(frame)
        ) if frame > 0 else 0.0

        ax.set_title(
            f"{title}\nstep {frame}/{n_steps}   cost so far: {running_cost:.3f}",
            fontsize=9, pad=4,
        )

    anim = animation.FuncAnimation(
        fig, draw_frame,
        frames=len(tour),
        interval=1000 // fps,
        blit=False,
    )

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(filepath), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return filepath
