# Portfolio updates

The Streamlit app builds on the original research with 2 modelling corrections:

- **ARIMA:** forecasts are aligned with the following month’s hotspot status.
- **XGBoost:** the classification threshold is selected using held-out validation predictions before the final model is trained.

The app’s results reflect these changes, therefore some figures differ from the original research outputs. Both versions are included separately in this repository.

The app also adds interactive maps, area-level incident histories and model comparisons. Evaluation uses the same 8 test months as the original research.