

from __future__ import annotations

import copy
import html
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
from jinja2 import Template
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


##############################################################################################
##############################################################################################


#Project paths 
ROOT = Path(__file__).resolve().parent
#Styles
COLORS = {
    "Correct hotspot": "#DF7064",
    "Missed hotspot": "#F2B6AE",
    "False alarm": "#8996A5",
    "Correct non-hotspot": "#81D7BA",
}
MODEL_COLORS = {
    "Persistence": "#996440",
    "Logistic regression": "#C1A766",
    "ARIMA": "#A1759D",
    "XGBoost": "#4FC4D3D7",
}
SCOPES = ["All areas", "Near cutoff (ranks 26–46)", "Status changes"]

##############################################################################################
##############################################################################################

#Page configuration and styles 
st.set_page_config(page_title="Cleveland |  Crime hotspot explorer", layout="wide")
st.markdown(
    """<style>
/*layout, headings, introductory text*/
.block-container {
    padding-top: 4.5rem;
    padding-bottom: 2.5rem;
    max-width: 1550px;
}
h1, h2, h3 {color: #E8EEF7; letter-spacing: -.035em;}
.eyebrow {
    color: #D6D93B;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.intro {
    font-size: 17px;
    color: #B7C6D9;
    max-width: 850px;
    line-height: 1.6;
    margin-bottom: 20px;
}

/*Sidebar background and metrics */
[data-testid="stSidebar"] {
    background: #131F33;
    border-right: 1px solid #2B3B52;
}
[data-testid="stMetric"] {
    background: #19263B;
    border: 1px solid #D6D93B;
    border-radius: 12px;
    padding: 10px 16px;
}
[data-testid="stMetricLabel"] {color: #B7C6D9;}
[data-testid="stMetricValue"] {color: #E8EEF7; font-size: 30px;}

/*Prediction-outcome summary above the map*/
.outcome-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 20px;
    align-items: center;
    background: #19263B;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 12px 0;
}

.outcome-item {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    color: #B7C6D9;
    font-size: 14px;

}

.outcome-item > span:nth-child(2) {
    white-space: nowrap;
}

.outcome-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

.outcome-count {
    color: #E8EEF7;
    font-size: 20px;
    font-weight: 600;
    margin-left: 4px;

}

@media (max-width: 700px) {
    .outcome-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .outcome-item > span:nth-child(2) {
        white-space: normal;
    }
}

.map-caption {
    text-align: center;
    font-size: 16px;
    color: #B7C6D9;
    margin: 8px 0;
}

/*Selected LSOA heading and details below the map */
.area-heading {display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px;margin:18px 0 12px;color:#E8EEF7;font-size:20px;}
.area-code {color:#B7C6D9;font-size:13px;}
.area-details {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;background:#19263B;border-radius:10px;padding:16px 20px;margin-bottom:12px;}
.area-detail {display:flex;flex-direction:column;gap:6px;}
.area-detail span {color:#B7C6D9;font-size:13px;}
.area-detail strong {color:#E8EEF7;font-size:20px;font-weight:600;}
@media(max-width:700px) {.area-details {grid-template-columns:repeat(2,minmax(0,1fr));}}

/*Model comparison table header */
[data-testid="stTable"] thead th {
    background-color: #263A52 !important;
    color: #E8EEF7 !important;
}

/* Navigation button */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    min-height: 46px;
    border-radius: 8px;
    justify-content: flex-start !important;
    text-align: left !important;
    padding-left: 18px !important;
}

/*Aligning the label container inside the button */
[data-testid="stSidebar"] [data-testid="stButton"] button > div {
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
}

/*Selected navigation button */
[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #263A52 !important;
    color: #D6D93B !important;
    border: 1px solid transparent !important;
}

/*Keeping the selected label and icon yellow */
[data-testid="stSidebar"] button[kind="primary"] * {
    color: #D6D93B !important;
}

/*Unselected navigation buttons */
[data-testid="stSidebar"] button[kind="secondary"] {
    background-color: transparent !important;
    color: #B7C6D9 !important;
    border: 1px solid transparent !important;
}

/*Hover over an unselected button */
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background-color: #19263B !important;
    color: #E8EEF7 !important;
    border-color: #526B8A !important;
}

/*Sidebar project summary and section labels*/
.sidebar-summary {
    color: #B7C6D9;
    margin-top: 12px;
}

.sidebar-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: #B7C6D9;
    margin-bottom: 8px;
}

.sidebar-summary p {
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 22px;
}


</style>""",
    unsafe_allow_html=True,
)

##############################################################################################
##############################################################################################

#Loading & caching data
@st.cache_data
def load_data():
    p = pd.read_csv(ROOT / "data/predictions_portfolio_v1.csv")
    h = pd.read_csv(ROOT / "data/crime_history.csv")
    m = pd.read_csv(ROOT / "data/metrics.csv")
    geo = json.loads((ROOT / "data/lsoa_boundaries.geojson").read_text())
    return p, h, m[m.run.eq("portfolio_v1")].copy(), geo

#Converting date into month & year
def month_label(value):
    return pd.Period(value, freq="M").strftime("%B %Y")

#Keeping the rows matching the chosen exploration or eva group
def select_scope(frame, scope):
    if scope == SCOPES[1]:
        return frame[frame.boundary_zone].copy()
    if scope == SCOPES[2]:
        return frame[frame.is_transition].copy()
    return frame.copy()


#chart & plot style
def format_chart(fig, height=320):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=15, t=35, b=10),
        font=dict(color="#B7C6D9"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    return fig

#Crime history chart
def history_chart(history, code, target_month):
    #Showing one LSOA history and marking the selected prediction month
    rows = history[history.lsoa_code.eq(code)].copy()
    rows["date"] = pd.to_datetime(rows.month)
    fig = go.Figure(
        go.Scatter(
            x=rows.date,
            y=rows.crime_count,
            mode="lines",
            line=dict(color="#D6D93B", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(22,133,119,.07)",
            name="Recorded incidents",
        )
    )
    chosen = rows[rows.month.eq(target_month)]
    fig.add_trace(
        go.Scatter(
            x=chosen.date,
            y=chosen.crime_count,
            mode="markers",
            marker=dict(size=10, color="#26D8C9"),
            name="Selected target month",
        )
    )
    fig.update_yaxes(title="Recorded incidents", rangemode="tozero")
    fig.update_layout(showlegend=False)
    fig.update_traces(
        hovertemplate=(
            "<b>%{x|%B %Y}</b><br>"
            "Recorded incidents: <b>%{y:,.0f}</b>"
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor="#19263B",
            bordercolor="#526B8A",
            font=dict(color="#E8EEF7", size=14),
        ),
    )
    return format_chart(fig, 245)

##############################################################################################
##############################################################################################

#Cutting map tiles outside of study area before display 
class StudyAreaTiles(folium.MacroElement):

    #Drawing only study area pixels into each tile, including during animations.

    _template = Template(r"""
    {% macro script(this, kwargs) %}
    (function () {
        const map = {{ this._parent.get_name() }};
        const polygons = {{ this.polygons | tojson }};
        const studyBounds = L.latLngBounds({{ this.bounds | tojson }});
        map.getContainer().style.background = '#0E182A';

        //Caching projected outline coordinates once per integer tile zoom
        const projected = new Map();
        const layer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            bounds: studyBounds,
            noWrap: true,
            maxZoom: 19,
            updateWhenIdle: true,
            keepBuffer: 2
        });
        layer.createTile = function (coords, done) {
            const tile = document.createElement('canvas');
            const size = this.getTileSize();
            tile.width = size.x;
            tile.height = size.y;
            const ctx = tile.getContext('2d');
            if (!projected.has(coords.z)) {
                projected.set(coords.z, polygons.map(polygon => polygon.map(ring =>
                    ring.map(point => map.project(L.latLng(point[1], point[0]), coords.z))
                )));
            }
            const originX = coords.x * size.x;
            const originY = coords.y * size.y;
            ctx.beginPath();
            projected.get(coords.z).forEach(polygon => polygon.forEach(ring => {
                ring.forEach((point, index) => {
                    const x = point.x - originX;
                    const y = point.y - originY;
                    if (index === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.closePath();
            }));
            //Even-odd clipping to preserve islands and holes in the outline
            ctx.clip('evenodd');
            const img = new Image();
            img.onload = function () {
                ctx.drawImage(img, 0, 0, size.x, size.y);
                done(null, tile);
            };
            img.onerror = function () {
                //Don't substitute an unclipped tile if a download fails
                done(new Error('Street-map tile unavailable'), tile);
            };
            img.src = this.getTileUrl(coords);
            return tile;
        };
        layer.addTo(map);

        
        //Keeping the whole study area available even when filtering to one borough
        function updateZoomLimit(initial) {
            const size = map.getSize();
            if (size.x < 1 || size.y < 1) return;
            const fitZoom = map.getBoundsZoom(studyBounds, false, L.point(28, 28));
            map.setMinZoom(fitZoom);
            map.setMaxBounds(studyBounds.pad(0.08));
            if (initial) map.fitBounds(studyBounds, {padding: [14, 14], animate: false});
        }
        map.whenReady(function () { updateZoomLimit(true); });
        map.on('resize', function () { updateZoomLimit(false); });
    })();
    {% endmacro %}
    """)


    #Creating study area boundary
    def __init__(self, outline):
        super().__init__()
        self._name = "StudyAreaTiles"
        geometry = mapping(outline)
        self.polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        west, south, east, north = outline.bounds
        self.bounds = [[south, west], [north, east]]


@st.cache_data
def study_outline(geo):
    #Using every supplied study polygon, irrespective of display filters
    return unary_union([shape(feature["geometry"]) for feature in geo["features"]])


#Building the map for the current filters
def map_for(frame, geo, context_codes):
    #Attaching results using area codes
    lookup = frame.set_index("lsoa_code")
    active = []
    context = []
    for feature in geo["features"]:
        code = feature["properties"]["lsoa_code"]
        if code not in context_codes:
            continue
        obj = copy.deepcopy(feature)
        if code not in lookup.index:
            context.append(obj)
            continue
        row = lookup.loc[code]
        #Hover labels with only the name and outcome
        obj["properties"].update(area=row.lsoa_name, outcome=row.outcome)
        active.append(obj)

    #Creating base map
    map_obj = folium.Map(
        location=[54.58, -1.22],
        zoom_start=10,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
        zoom_snap=0.25,
        zoom_delta=0.5,
        max_zoom=19,
        max_bounds_viscosity=1.0,
        bounce_at_zoom_limits=False,
    )

    #Adding custom street-map layer
    StudyAreaTiles(study_outline(geo)).add_to(map_obj)

    #Drawing the background areas
    if context:
        folium.GeoJson(
            dict(type="FeatureCollection", features=context),
            style_function=lambda _: dict(
                fillColor="#EDF1F4", color="#C8D2DA", weight=0.5, fillOpacity=0.35
            ),
            name="Other study areas",
            #to not respond to clicks/hover
            interactive=False,
        ).add_to(map_obj)

    #Drawing the prediction outcomes
    if active:
        layer = folium.GeoJson(
            dict(type="FeatureCollection", features=active),
            name="Prediction outcomes",
            style_function=lambda f: dict(
                fillColor=COLORS[f["properties"]["outcome"]],
                color="#FFFFFF",
                weight=0.8,
                fillOpacity=0.65,
            ),
            highlight_function=lambda _: dict(
                weight=2, color="#203A4F", fillOpacity=0.95
            ),
            #tooltip
            tooltip=folium.GeoJsonTooltip(
                fields=["area", "outcome"], aliases=["Area", "Result"], sticky=False
            ),
        ).add_to(map_obj)
        map_obj.fit_bounds(layer.get_bounds(), padding=(14, 14))
    return map_obj

##############################################################################################
##############################################################################################


#Loading the app data before building navigation and page content
try:
    predictions, history, metrics, boundaries = load_data()
except FileNotFoundError:
    st.error(
        "data folder is missing."
    )
    st.stop()

#Sidebar navigation: retain the selected page across reruns
if "active_page" not in st.session_state:
    st.session_state.active_page = "Prediction map"

#Updating navigation before rendering the next page
def change_page(destination):
    st.session_state.active_page = destination


with st.sidebar:
    st.markdown(
        """
        <div style="
            color: #B7C6D9;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 1.5px;
            margin-bottom: 20px;
        ">
            PROJECT NAVIGATION
        </div>
        """,
        unsafe_allow_html=True,
    )

    navigation = [
        ("Prediction map", "Prediction map", ":material/map:"),
        ("Model comparison", "Model comparison", ":material/bar_chart:"),
        ("About", "About & methods", ":material/info:"),
    ]

    for label, destination, icon in navigation:
        st.button(
            label,
            icon=icon,
            key=f"nav_{destination}",
            type=(
                "primary"
                if st.session_state.active_page == destination
                else "secondary"
            ),
            width="stretch",
            on_click=change_page,
            args=(destination,),
        )

    st.divider()
    st.markdown(
        """
    <div class="sidebar-summary">
    <div class="sidebar-section-label">AT A GLANCE</div>
    <p>359 LSOAs | 8 test months</p>
    <div class="sidebar-section-label">MODELS</div>
    <p>XGBoost | ARIMA<br>Logistic regression | Persistence</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<div class="sidebar-section-label">PROJECT LINKS</div>',
        unsafe_allow_html=True,
    )

    st.markdown("[More projects ↗](https://github.com/edit-b)")

page = st.session_state.active_page


##############################################################################################
##############################################################################################


###################################### PAGE 1: monthly predictions ###########################
if page == "Prediction map":
    st.markdown(
        '<div class="eyebrow">MONTHLY HOTSPOT PREDICTIONS</div>', unsafe_allow_html=True
    )
    st.title("Cleveland crime hotspot explorer")
    st.markdown(
        '<div class="intro">Explore monthly forecasts across 359 neighbourhoods and see how each model performed.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    #Blocking the area filters while a specific LSOA is selected
    area_is_selected = st.session_state.get("area_selection") is not None
    #controls = st.columns([1.05, 1.05, 1.3, 1.4, 1.8])
    
    # Give longer selections enough space by arranging filters in two rows.
    top_row = st.columns([1, 1.2, 1.5])
    bottom_row = st.columns([1, 1.5])

    controls = [*top_row, *bottom_row]
    with controls[0]:
        month = st.selectbox(
            "Predicted month",
            sorted(predictions.target_month.unique()),
            format_func=month_label,
        )
    with controls[1]:
        model = st.selectbox(
            "Model", ["XGBoost", "Persistence", "Logistic regression", "ARIMA"]
        )
    with controls[2]:
        authority = st.selectbox(
            "Local authority",
            ["All authorities"] + sorted(predictions.local_authority.unique()),
            disabled=area_is_selected,
        )
    with controls[3]:
        scope = st.selectbox(
            "Areas to explore",
            SCOPES,
            disabled=area_is_selected,
        )


    #Applying month/model/authority first then the exploration group
    full = predictions[
        predictions.target_month.eq(month) & predictions.model.eq(model)
    ].copy()
    if authority != "All authorities":
        full = full[full.local_authority.eq(authority)]
    selected = select_scope(full, scope)
    if scope == SCOPES[2]:
        st.caption("Status changes are identified after the outcome is known.")
    st.caption(
        "A hotspot is one of the 36 LSOAs with the highest recorded crime counts in a given month."
    )


    #Offering only matching LSOAs, clearing any selection that became invalid
    options = selected.sort_values("lsoa_name").lsoa_code.tolist()
    names = selected.set_index("lsoa_code").lsoa_name.to_dict()
    previous_code = st.session_state.get("area_selection")
    selection_cleared = previous_code is not None and previous_code not in options
    if previous_code not in options:
        st.session_state["area_selection"] = None
    with controls[4]:
        code = st.selectbox(
            "Select LSOA",
            options,
            index=None,
            placeholder="Choose a matching LSOA",
            disabled=not options,
            format_func=lambda c: f"{names[c]} · {c}",
            key="area_selection",
        )

    if selection_cleared:
        st.caption(
            "Your previous LSOA does not match these filters. Choose another LSOA or adjust the filters."
        )
    if selected.empty:
        st.info(
            "No LSOAs match this combination. Choose another month, authority or area filter."
        )
        st.stop()

    
    #Build a map identifier from the current filters
    ## Changing a filter creates a new map, selecting an LSOA keeps the same map
    signature = f"{month}|{model}|{authority}|{scope}"
    map_key = "map_area_details_" + signature

    #Updating the LSOA dropdown when a highlighted area is clicked
    def select_area_from_map():
        ##Reading the information returned by this map
        event = st.session_state.get(map_key) or {}
         #Getting the clicked map feature (the LSOA polygon)
        drawing = event.get("last_active_drawing") or {}
        #Extracting LSOA code
        #Only accepting areas available under current filters (runs before dropdown redrawn)
        clicked = drawing.get("properties", {}).get("lsoa_code")
        if clicked in options:
            st.session_state["area_selection"] = clicked

    #Counting the highlighted LSOas in each prediction-outcome category
    counts = selected.outcome.value_counts()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="map-caption">{len(selected)} LSOAs highlighted. '
        "Choose an LSOA above or click the map for details below.</div>",
        unsafe_allow_html=True,
    )
    strip_items = "".join(
        f'<div class="outcome-item">'
        f'<span class="outcome-dot" style="background:{color}"></span>'
        f"<span>{name}</span>"
        f'<span class="outcome-count">{int(counts.get(name, 0))}</span>'
        f"</div>"
        for name, color in COLORS.items()
    )

    st.markdown(
        f'<div class="outcome-strip">{strip_items}</div>',
        unsafe_allow_html=True,
    )


    #DRawing the map
    map_obj = map_for(selected, boundaries, set(full.lsoa_code))
    #Creating a separate layer for the selected area
    selection_outline = folium.FeatureGroup(name="Selected LSOA")
    #Finding and outlining selected area
    if code is not None:
        feature = next(
            f for f in boundaries["features"] if f["properties"]["lsoa_code"] == code
        )
        #Drawing the polygon's outline
        folium.GeoJson(
            feature,
            style_function=lambda _: dict(
                color="#19263B", weight=3, fill=False, opacity=1
            ),
            interactive=False,
        ).add_to(selection_outline)

    #Displaying the map, handling interacions
    st_folium(
        map_obj,
        height=450,
        use_container_width=True,
        key=map_key,
        returned_objects=["last_active_drawing"],
        feature_group_to_add=selection_outline,
        on_change=select_area_from_map,
    )
    st.caption(
        "Boundaries: Office for National Statistics, Open Government Licence v3.0. Contains OS data © Crown copyright and database right 2026."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    #Details from the same filtered rows as the map and dropdown
    if code is None:
        st.info(
            "Choose an LSOA from the dropdown or click a highlighted area to view its details."
        )
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        area = selected[selected.lsoa_code.eq(code)].iloc[0]

        st.markdown(
            f'<div class="area-heading"><strong>{html.escape(area.lsoa_name)}</strong>'
            f'<span class="area-code">{html.escape(code)}</span></div>',
            unsafe_allow_html=True,
        )

        details = [
            ("Prediction", "Hotspot" if area.predicted_hotspot else "Non-hotspot"),
            ("Actual status", "Hotspot" if area.actual_hotspot else "Non-hotspot"),
            (
                f"Incidents · {month_label(area.observation_month)}",
                str(int(area.current_crime_count)),
            ),
            (f"Incidents · {month_label(month)}", str(int(area.actual_crime_count))),
        ]
        detail_items = "".join(
            f'<div class="area-detail"><span>{html.escape(label)}</span>'
            f"<strong>{html.escape(value)}</strong></div>"
            for label, value in details
        )
        st.markdown(
            f'<div class="area-details">{detail_items}</div>', unsafe_allow_html=True
        )
        if pd.notna(area.probability):
            st.caption(
                f"Model score: {area.probability:.1%} · Decision threshold: {area.decision_threshold:.3f}. Scores are uncalibrated, they are not validated risk probabilities."
            )
        else:
            st.caption("This model does not produce a hotspot probability.")
        st.plotly_chart(history_chart(history, code, month), width="stretch")
        st.caption(
            "The chart includes later observed months for historical context. Later observations were not used to make the selected forecast."
        )
        st.markdown("<br>", unsafe_allow_html=True)

    #Exporting the results and all source columns in the CSV
    with st.expander("View and download the selected results"):
        st.markdown("<br>", unsafe_allow_html=True)
        cols = [
            "lsoa_code",
            "lsoa_name",
            "target_month",
            "actual_hotspot",
            "predicted_hotspot",
            "outcome",
            "probability",
        ]
        st.dataframe(selected[cols], hide_index=True, width="stretch")
        st.download_button(
            "Download selected results (CSV)",
            selected.to_csv(index=False),
            file_name=f"cleveland_{month}_{model.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


##############################################################################################
##############################################################################################

################### PAGE 2: Model comparison and deprivation group errors ####################

elif page == "Model comparison":
    st.markdown('<div class="eyebrow">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    st.title("Comparing hotspot prediction models")
    st.markdown(
        '<div class="intro">Most areas keep the same hotspot status. A strong overall accuracy can hide the harder cases near the cutoff.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    #evaluation selector
    scope = st.radio(
        "Evaluation group", SCOPES + ["Outside cutoff zone"], horizontal=True
    )
    view = metrics[metrics.scope.eq(scope)].copy()
    baseline = view[view.model.eq("Persistence")].iloc[0]
    xgb = view[view.model.eq("XGBoost")].iloc[0]
    #Metric cards
    a, b, c = st.columns(3)
    a.metric("Area-month observations", int(xgb.n))
    b.metric("XGBoost accuracy", f"{xgb.accuracy:.1%}")
    c.metric(
        "Difference from persistence",
        f"{100 * (xgb.accuracy - baseline.accuracy):+.1f} pp",
        help="Difference in accuracy, measured in percentage points.",
    )
    if scope == SCOPES[1]:
        st.caption(
            "Ranks 26-46 in the observation month: 21 areas per month around rank 36."
        )
    elif scope == SCOPES[2]:
        st.caption(
            "These areas changed status between the observation and target months. Because Persistence predicts no change, it cannot correctly predict these cases."
        )


    #Reshaping model metrics into one row per model/metric for the grouped bars
    melted = view.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="Metric",
        value_name="Score",
    )
    #Drawing figure
    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.plotly_chart(format_chart(fig, 380), width="stretch")

    #DRawing table
    st.table(
        view[["model", "n", "accuracy", "precision", "recall", "f1", "auc"]]
        .set_index("model")
        .style.format(
            {k: "{:.3f}" for k in ["accuracy", "precision", "recall", "f1", "auc"]},
            na_rep="—",
        )
    )
    st.caption(
        "Precision: how many flagged areas were hotspots. Recall: how many actual hotspots were found. F1 balances precision and recall. Zero-denominator metrics are reported as 0. AUC is available only for models with continuous hotspot scores."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    #XGBoost Deprivation plot
    st.subheader("False alarms and deprivation")
    st.write(
        "Inspect XGBoost's false-positive rate across 3 deprivation groups. This highlights differences in model performance and helps identify where false alarms are more common."
    )
    
    x = predictions[predictions.model.eq("XGBoost")].copy()
    areas = x[["lsoa_code", "imd_decile"]].drop_duplicates().sort_values("lsoa_code")

    #Matching the original audit's equal area grouping 
    areas["deprivation_severity"] = 11 - areas.imd_decile
    areas["group"] = pd.qcut(
        areas.deprivation_severity.rank(method="first"),
        3,
        labels=["Lower deprivation", "Middle group", "Higher deprivation"],
    )
    x = x.merge(areas[["lsoa_code", "group"]], on="lsoa_code", validate="many_to_one")
    x["fp"] = (x.predicted_hotspot.eq(1) & x.actual_hotspot.eq(0)).astype(int)
    x["non_hotspot"] = x.actual_hotspot.eq(0).astype(int)
    audit = (
        x.groupby("group", observed=True)
        .agg(false_alarms=("fp", "sum"), actual_non_hotspots=("non_hotspot", "sum"))
        .reset_index()
    )
    # False-positive rate = false alarms / actual non-hotspot observations
    audit["false_positive_rate"] = (
        audit.false_alarms / audit.actual_non_hotspots.replace(0, np.nan)
    )


    ##Drawing scatter plot
    fig = px.scatter(
        audit,
        x="false_positive_rate",
        y="group",
        text="false_positive_rate",
        custom_data=["false_alarms", "actual_non_hotspots"],
    )

    fig = format_chart(fig, 250)

    fig.update_traces(
        marker=dict(size=13, color="#E08967"),
        texttemplate="%{x:.1%}",
        textposition="middle right",
        textfont=dict(size=14, color="#E8EEF7"),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "False-positive rate: %{x:.1%}<br>"
            "False alarms: %{customdata[0]:,.0f}<br>"
            "Actual non-hotspot observations: %{customdata[1]:,.0f}"
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor="#19263B",
            bordercolor="#526B8A",
            font=dict(color="#E8EEF7", size=14),
        ),
    )

    fig.update_xaxes(
        title="False-positive rate",
        tickformat=".0%",
        range=[0, max(0.05, audit["false_positive_rate"].max() * 1.25)],
        showgrid=True,
        gridcolor="rgba(183, 198, 217, 0.12)",
        zeroline=False,
    )

    fig.update_yaxes(
        title=None,
        categoryorder="array",
        categoryarray=[
            "Lower deprivation",
            "Middle group",
            "Higher deprivation",
        ],
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        ticks="",
        automargin=True,
    )

    fig.update_layout(
        margin=dict(l=15, r=45, t=15, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Based on all 359 LSOAs across the 8 test months. LSOAs are grouped by their level of deprivation. Changing the evaluation group above does not affect this chart."
    )


##############################################################################################
##############################################################################################

################### PAGE 3: About ############################################################


else:
    st.markdown('<div class="eyebrow">PROJECT BACKGROUND</div>', unsafe_allow_html=True)
    st.title("About the project")

    st.subheader("The question")
    st.write(
        "Can models identify which small areas will be crime hotspots next month, and do they improve on simply assuming that today's hotspots will stay hotspots?"
    )
    st.write(
        "A Lower-layer Super Output Area (LSOA) is a small statistical area containing approximately 1,000–3,000 residents."
        " This project covers 359 LSOAs in the Cleveland Police force area. Each month, the 36 LSOAs with the highest recorded incident counts are classified as hotspots. Incident counts include recorded crime and anti-social behaviour. "
        "Hotspot rankings use the total number of recorded incidents in each LSOA, without adjusting for population size."
    )
    st.subheader("Data and evaluation")
    st.markdown("""
- **Study area:** 359 modelled LSOAs across the four Teesside local authorities, excluded codes are not shown.
- **Crime history:** April 2023-March 2026, from the supplied Cleveland Police open-data extract.
- **Training rows:** July 2023-June 2025, predicting August 2023-July 2025.
- **Test rows:** July 2025-February 2026, predicting August 2025-March 2026.
- **Models:** persistence, class-weighted logistic regression, per-area ARIMA(1,0,1), tuned XGBoost.
- **Inputs:** crime history, neighbouring-area history, seasonal features, weather, holidays, deprivation and local authority.
""")

    st.subheader("Understanding the results")
    st.markdown("""
- The models were evaluated using historical data. Their performance may change when applied to future months.
- Results depend on which incidents are reported to and recorded by the police.
- Hotspots are based on monthly rankings. An LSOA can move into or out of the top 36 as incident counts change across the study area.
- False-alarm rates vary across deprivation groups. Further analysis would be needed to understand what drives these differences.

""")
    st.subheader("Sources and reproducibility")
    st.markdown(
        "Crime: [data.police.uk](https://data.police.uk/). Boundaries: [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/). Full source notes and commands are in `docs/METHODOLOGY.md` and `README.md` in the project download."
    )
    st.caption(
        "Boundaries supplied with the project: ONS LSOA 2021 BGC V5. Source: Office for National Statistics, Open Government Licence v3.0. Contains OS data © Crown copyright and database right 2026. Map tiles © OpenStreetMap contributors."
    )
