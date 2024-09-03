# Introduction

## Structure

```bash
.
├── README.md
├── app
│   ├── __init__.py
│   ├── components.py
│   ├── p0.py
│   ├── p1.py
│   ├── p2.py
│   └── p3.py
├── assets
│   └── doe.svg
├── data
│   ├── foo.xlsx
│   └── bar.xlsx
├── main.py
├── requirements.txt
├── styles.css
└── utils
    ├── __init__.py
    ├── custom_regression.py
    ├── data.py
    ├── math.py
    ├── model.py
    ├── models.py
    ├── plots.py
    └── regression.py
```

## Files

### `main.py`

```python
import app

import utils
from utils import regression, models, custom_regression, plots
# Not the best way to import, but it's convenient. Please make this nicer

import gc
import glob

import streamlit as st

def main():
    st.set_page_config(
        layout="wide",
        page_title=app.TITLE,
    )
    
    st.markdown(f"""
    <style>
    {utils.get_styles()}
    </style>
    """, unsafe_allow_html=True)
    
    gc.set_threshold(0) # disable garbage collection
    app.main()
    gc.collect() # collect manually after every execution, to avoid memory issues

if __name__ == "__main__":
    main()
```

### `styles.css`

```css
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}
[data-testid="stHeader"]
{
    display: none !important
}
[data-testid="stSidebarHeader"] {
    position: absolute;
    top: 0;
    right: 0;
    z-index: 10;
}
```

### `utils/__init__.py`

```python
@st.cache_data(ttl=60*60)
def get_styles():
    with open("styles.css", "r") as f:
        styles = f.read()
    return styles


def get_page_title(menu_selected_external, menu_selected_internal):
    header = (
        menu_selected_external +
        (": " + menu_selected_internal if menu_selected_internal is not None else "")
    )
    st.subheader(header)
    

def internal_navigation(menu_selected_external, menu_options_external):
    # with st.sidebar:
    #     st.divider()

    menu_options_internal_mappings = {
        0: [
            
        ],
        1: [
            "Foo",
            "Bar"
        ],
    }
    menu_options_internal = (
        menu_options_internal_mappings
        .get(menu_options_external.index(menu_selected_external), [])
    )
    # menu_selected_internal = st.radio(
    #     label="Sub Menu",
    #     label_visibility = "visible",
    #     options=menu_options_internal
    # )
    
    menu_selected_internal = option_menu( # st.radio
        menu_title = None, # "Menu",
        options = menu_options_internal,
        orientation = "horizontal",
        styles = {
            "container": {"margin": "0 !important", "padding": "0 !important"}, # , "background": "none"
            "nav": {"font-size": "0.75em"},
            "icon": {"display": "none"},
            "nav-link": {"margin":"0", "padding": "0.5ex 1.5ex"},
        },
    )
        
    return menu_selected_internal, menu_options_internal
```

### `app/__init__.py`

```python
import utils
from utils import data, regression, custom_regression, models, plots

from app import (
    components,
    p0,
    p1,
    p2,
    p3
)

import streamlit as st
from streamlit_option_menu import option_menu

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    mean_absolute_percentage_error as mape,
    # mean_squared_error as mse,
    root_mean_squared_error as rmse,
    mean_absolute_error as mae,
    r2_score as r2
)

import numpy as np
import pandas as pd
import polars as pl

TITLE = "Chemical Kinetics Modelling"

def main():
    data = utils.data # why???

    menu_options_external = [
        "Home",
	      "Foo",
        "Bar",
    ]
    
    with st.sidebar:
        st.title(TITLE)
        # menu_selected_external = st.radio(
        #     "Menu",
        #     menu_options_external,
        #     label_visibility = "collapsed"
        # )
    menu_selected_external = option_menu( # st.radio
        menu_title = None, # "Menu",
        options = menu_options_external,
        orientation = "horizontal",
        styles = {
            "container": {"margin": "0 !important", "padding": "0 !important",},  # "background": "none"
            "nav": {"font-size": "0.75em"},
            "icon": {"display": "none"},
            "nav-link": {"margin":"0", "padding": "0.5ex 1.5ex"},
        }
        # label_visibility = "collapsed"
    )

    if menu_selected_external == menu_options_external[0]:
        p0.main()

    menu_selected_internal, menu_options_internal = utils.internal_navigation(menu_selected_external, menu_options_external)

    utils.get_page_title(menu_selected_external, menu_selected_internal)

    if menu_selected_external == menu_options_external[1]:
        p1.main(menu_selected_internal, menu_options_internal)

    if menu_selected_external == menu_options_external[2]:
        p2.main(menu_selected_internal, menu_options_internal, df)
```

### `p1.py`

```python
from utils import data
import streamlit as st

def main(menu_selected_internal, menu_options_internal):
    if menu_selected_internal == menu_options_internal[0]:
        st.dataframe(
            data.get_details().collect(),
            use_container_width=True,
            hide_index=True
        )
    elif menu_selected_internal == menu_options_internal[1]:
        st.dataframe(
            data.get_readings().collect(),
            use_container_width=True,
            hide_index=True
        )
    st.stop()
```

### `app/components.py`

```python
import streamlit as st
import utils

def input_filters(df, menu_selected_external, menu_options_external, menu_selected_internal, menu_options_internal):
    with st.sidebar:
        # st.divider()
        st.subheader("Data Input Filters")
        
        filter_cols = ["Study_Identifier", "Temperature"] # , "Sample_Identifier"
        filters_selected = {}
        
        n_cols = 3
        cols = st.columns([2, 1])
        current_col = 0
        for col in filter_cols:
            comparison_page == (menu_selected_external==menu_options_external[3] and menu_selected_internal==menu_options_external[2])
            
            with cols[current_col]:
                if (
                    (col == filter_cols[0] and not comparison_page)
                    or
                    (col == filter_cols[1] and comparison_page)
                ):
                    single_only = True
                else:
                    single_only = False
                    
                filters_selected.update({
                    col: generate_filter(df, col, single_only)
                })
                current_col = (current_col+1)%n_cols
            
    keys = list(filters_selected.keys())
    values = list(filters_selected.values())
    
    return keys, values

def generate_filter(df, col, single_only=False):
    """
    returns list for modularity and ease
    """
    
    options = generate_options(df, col)
    
    if single_only:
        selected = [st.selectbox(
            label = col.split("_")[0],
            options = options
        )]
    else:
        selected = st.multiselect(
            label = col.split("_")[0],
            options = options
        )
    if len(selected) == 0:
        selected = options

    return selected
```

