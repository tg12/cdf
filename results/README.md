# Results

Generated dashboards and local analysis artifacts are written to this directory.

The default output path is `results/consolidation_dashboard_latest.html`.

Included artifact:

- [`consolidation_dashboard_latest.html`](consolidation_dashboard_latest.html): example interactive dashboard generated from the current local workflow

GitHub will display the HTML source rather than the rendered interactive page. Open the file locally in a browser to inspect the dashboard.

To regenerate the artifact with the bundled sample dataset:

```bash
python3 cdf.py
```

Track documentation here if needed. Treat HTML dashboards and exported tables as reproducible artifacts derived from local data and configuration.
