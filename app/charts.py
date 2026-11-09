"""Small chart helpers shared by the pages, so every bar chart looks the same."""

import plotly.graph_objects as go

from data import NEUTRAL


def hbar(labels, values, title, value_format=",.0f", hover_label="Postings",
         colours=None, height=None, x_title=None):
    """Horizontal bar chart, largest at the top.

    Horizontal because the labels (skill names, job titles) are long and
    would have to be slanted under vertical bars.
    """
    labels, values = list(labels), list(values)
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colours or NEUTRAL, line=dict(width=0)),
        text=[format(v, value_format) for v in values], textposition="outside",
        cliponaxis=False,
        hovertemplate=f"<b>%{{y}}</b><br>{hover_label}: %{{x:{value_format}}}<extra></extra>",
    ))
    fig.update_layout(
        title=title, template="plotly_white",
        height=height or max(260, 28 * len(labels) + 90),
        margin=dict(l=10, r=40, t=50, b=30), bargap=0.35,
        yaxis=dict(autorange="reversed", automargin=True),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#eeeeee", zeroline=False,
                   tickformat=value_format),
    )
    return fig
