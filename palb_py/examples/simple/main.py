import numpy as np
import palb_py as palb
import plotly.graph_objects as go


def main():
    xs = np.linspace(-1, 1, 100_000)
    ys_true = -4 * xs - 2
    ys_noisy = ys_true + np.random.default_rng().laplace(0, 1, size=xs.shape)
    (slope, intercept) = palb.l1line_xy(xs, ys_noisy).to_slope_intercept()

    xs_plot = np.linspace(xs[0], xs[-1], num=2)
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=xs,
                y=ys_noisy,
                mode="markers",
                name="Noisy Data",
                marker=dict(size=2),
            ),
            go.Scattergl(
                x=xs_plot,
                y=slope * xs_plot + intercept,
                mode="lines",
                name="Predicted Line",
                line=dict(width=4),
            ),
            go.Scattergl(
                x=xs[[0, -1]],
                y=ys_true[[0, -1]],
                mode="lines",
                name="True Line",
                line=dict(width=4, dash="dash"),
            ),
        ]
    )
    fig.show()


if __name__ == "__main__":
    main()
