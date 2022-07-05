# MANDATORY DEPENDECIES
import os
import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

# SET PAGE LAYOUT TO WIDE
st.set_page_config(page_title="Open Source - PmDARM", layout="wide")

# Title on the Header of page
st.title('Open Source Product Analysis Dashboard with Association Rule Mining - PmDARM')

# Making containers
sidebar = st.sidebar.container()
header = st.container()
metric_col = st.container()
main_contents = st.container()
a_priori = st.container()

# Hot encode function for suitability of the dataframe


def hot_encode(x):
    if(x <= 0):
        return 0
    if(x >= 1):
        return 1

# # Primary Unjargonizer - for FSets - MAKE AD-HOC FUNCTION TO REMOVE UNRELATED WORDS (UN-JARGONIZER)


def removeJargon(sentence):
    new_str = ''
    for char in sentence:
        if char not in (',', '{', '}', '(', ')', '\'',):
            new_str += char
    new_str = new_str.replace('frozenset', '')
    return new_str

# FSets unjargonizer - MAKE AD-HOC FUNCTION TO CALL F-SET RELATED DATASET TO UN-JARGONIZE (UN-JARGONIZER BY DATASET AND COL-NAME)


def fSets_remove(dataset, colName):
    # MAKE NEW LIST TO STORE ALL P-W-B COLUMN FOR REPLACE LATER
    gotStr = []
    for i in range(len(dataset)):
        gotStr.append(removeJargon(
            str(dataset[colName][i])).replace(' ', ' and '))

    # REPLACE ALL P-W-B COLUMN INTO NEW STRING FROM PREVIOUS LIST
    dataset[colName] = dataset[colName].replace(
        dataset[colName].tolist(), gotStr)
    return dataset


def error_default():
    with st.expander("Oops... something went wrong."):
        st.markdown('''
            #### Looks like you've run into **one** *(or more)* possible error(s):
            - Your dataset is not the correct extension format. It only accepts *".csv"* extension files.
            
            - Your dataset columns may not be correctly named. It should follow the strict-naming convention as *(but not in order)*:
                - **[Order ID]**: This acts as the primary key for all purchases.
                - **[Order Date]**: This would be the datetime object for the dataframe.
                - **[State]**: This would be the participating states. It serves to refer from the indian geojson file.
                - **[City]**: The participating city for the states. 
                - **[Category]**: The category of items purchased.
                - **[Sub-Category]**: The subset of items from 'Category'.

            - Your dataset contains null values.

            - Your dataset elements may not be correlated with the column names.            
        ''')
        st.empty()


# SIDEBAR - TO INPUT RELEVANT DATASET
with sidebar:
    try:
        # Upload a file to the csv uploader
        data_file = st.file_uploader("Upload CSV", type=["csv"])
        if data_file is not None:

            # Uploaded file details
            file_details = {"filename": data_file.name,
                            "filetype": data_file.type, "filesize": data_file.size}

            # Caching dataset
            @st.cache(allow_output_mutation=True)
            def load_csv():
                dataset = pd.read_csv(data_file)
                return dataset
            dataset = load_csv()

            # Break if there is any NA values - Check for NaN values in each column
            for col in dataset.columns:
                NA_checker = dataset[col].isnull().values.any()
                if NA_checker == True:
                    with header:
                        st.caption("Error")
                        st.warning("{} has some null values in columns.".format(
                            data_file.name))
                        st.error("Unable to continue. Stopping process.")
                        error_default()
                    st.stop()

            # Dropping column name with streamlit's default indexer
            if dataset.columns[0] == "Unnamed: 0":
                dataset.drop('Unnamed: 0', inplace=True, axis=1)

            # Removing white trailing spaces from dataset according to remover_list
            remover_list = ['Order ID', 'State',
                            'City', 'Category', 'Sub-Category']
            for cols in remover_list:
                # st.text("Removing white space")
                dataset[cols] = dataset[cols].str.strip()

            # set "Order Date" to datetime format in python
            dataset["Order Date"] = pd.to_datetime(dataset["Order Date"])

        else:
            with header:
                st.caption("Error")
                st.text('Please specify valid dataset(s) in the sidebar')

    # File error
    except:
        st.error('File reading error.')
        with header:
            error_default()
        st.stop()

# SUCCESS DATASET RETRIEVAL - TOP HEADER
with header:
    try:
        # Preparing plots for DATE vs Quantity (bar and line)
        fig1 = px.bar(
            dataset,
            x=dataset["Order Date"].dt.strftime("%Y-%m").unique(),
            y=dataset.groupby(dataset["Order Date"].dt.strftime(
                "%Y-%m")).Quantity.sum(),
            labels={'x': 'Date', 'y': 'Quantity'},
            title="Date vs Quantity (bar)")
        fig2 = px.line(
            dataset,
            x=dataset["Order Date"].dt.strftime("%Y-%m").unique(),
            y=dataset.groupby(dataset["Order Date"].dt.strftime(
                "%Y-%m")).Quantity.sum(),
            labels={'x': 'Date', 'y': 'Quantity'},
            title="Date vs Quantity (line)")

        # Expander to show dataset
        with st.expander("See dataset"):
            st.write(dataset)

        # Header and prepare charts for DATE vs Quantity (bar and line)
        st.header("Monthly Sales (Quantity)")
        st.markdown('Over here it shows the monthly sales by **quantity** from {} to {}.'.format(
            dataset["Order Date"].dt.year.min(), dataset["Order Date"].dt.year.max()))
        QxM1, QxM2 = st.columns(2)
        QxM1.plotly_chart(fig1, use_container_width=True)
        QxM2.plotly_chart(fig2, use_container_width=True)

        # Expander to explore on the states purchases by year-month
        with st.expander("Show more details"):
            st.markdown(''' 
                > #### State Purchase Ranking             
                > Over here, we can see transactions of participating states over the fiscal year.''')

            # Making 2 columns
            choice_col, show_table_col = st.columns([1, 3])

            # Choice column
            with choice_col:
                time_choice = dataset["Order Date"].dt.strftime(
                    "%Y-%m").unique().tolist()
                selected_datetime = st.selectbox(
                    "Choose date (YYYY-mm):", options=time_choice, index=0)

            # Show selected choice column
            with show_table_col:
                rank_state = dataset.groupby([dataset["Order Date"].dt.strftime(
                    "%Y-%m"), "State"])["Order ID"].count().to_frame().reset_index()
                rank_state["Order Date"] = pd.to_datetime(
                    rank_state["Order Date"])
                table_df = rank_state[rank_state["Order Date"].dt.strftime(
                    "%Y-%m") == selected_datetime].sort_values('Order ID', ascending=False)
                table_df = table_df.rename(columns={'Order ID': 'Count'})
                st.write(table_df)

    except:
        # do nothing
        st.text("")

# Donut chart
with metric_col:
    metrics_1, metrics_2 = st.columns(2)
    # METRIC DATA 1
    with metrics_1:

        try:
            # Making new dataframe to store category sales by quantity
            catXquantity = dataset.groupby(dataset["Category"]).Quantity.sum(
            ).sort_values(ascending=False).to_frame().reset_index()

            # Preparing chart
            pieChart1 = px.pie(
                data_frame=catXquantity,
                values=catXquantity["Quantity"],
                names=catXquantity["Category"],
                color=catXquantity["Category"],
                title="Top Sales (Category)",
                template="presentation",
                hole=0.5
            )

            # Show chart
            st.plotly_chart(pieChart1, use_container_width=True)
        except:
            st.empty()

    # METRIC DATA 2
    with metrics_2:

        try:
            # Making new dataframe to store sub-category sales by quantity
            subCatXQuantity = dataset.groupby(["Sub-Category", "Category"]).Quantity.sum(
            ).sort_values(ascending=False).to_frame().reset_index()
            subCatXQuantity_forShow = subCatXQuantity.head(5)

            # Preparing chart
            pieChart2 = px.pie(
                data_frame=subCatXQuantity_forShow,
                values='Quantity',
                names='Sub-Category',
                color='Sub-Category',
                title="Top 5 Sales (Items)",
                template="presentation",
                labels={'Sub-Category': 'Item'},
                hole=0.5,
                hover_name='Category'
            )

            # Showing chart
            st.plotly_chart(pieChart2, use_container_width=True)
        except:
            st.empty()

    try:
        if dataset is not None:
            with st.expander("More details about Category and Sub-Category"):
                try:
                    catCol, itemCol = st.columns(2)
                    subCol1, subCol2, subCol3 = st.columns(3)
                    category = sorted(dataset["Category"].unique().tolist())
                    items = sorted(dataset["Sub-Category"].unique().tolist())
                    clothing = sorted(dataset.loc[dataset["Category"]
                                                  == "Clothing"]["Sub-Category"].unique().tolist())
                    electronics = sorted(
                        dataset.loc[dataset["Category"] == "Electronics"]["Sub-Category"].unique().tolist())
                    furniture = sorted(
                        dataset.loc[dataset["Category"] == "Furniture"]["Sub-Category"].unique().tolist())
                    with catCol:
                        st.markdown(' #### List of Categories.')
                        st.table(category)

                        with subCol1:
                            st.markdown(' **Clothing** ')
                            st.table(clothing)
                        with subCol2:
                            st.markdown(' **Electronics** ')
                            st.table(electronics)
                        with subCol3:
                            st.markdown(' **Furniture** ')
                            st.table(furniture)

                    with itemCol:
                        st.markdown(' #### List of Sub-Categories.')
                        st.table(items)
                except:
                    st.empty()
    except:
        st.empty()

# Thematic/Choropleth Map
with st.spinner('Coloring map... wait for it...'):
    with main_contents:
        try:
            # GETTING JSON FILE AND LOAD INTO "indian_states"
            indian_states = json.load(open('states_india.geojson'))

            # MAKING NEW FEATURE IN JSON AS ID FROM STATE CODE FOR DATASET LATER
            state_id_map = {}
            for feature in indian_states['features']:
                feature['id'] = feature['properties']['state_code']
                state_id_map[feature['properties']['st_nm']] = feature['id']

            # MAP ID WITH STATE (DATASET) FROM THE JSON FILE
            dataset['id'] = dataset['State'].apply(lambda x: state_id_map[x])

            # MAKING NEW DF FOR PURPOSE OF CHORO-MAP
            for_map = pd.DataFrame()
            for_map["State"] = sorted(dataset["State"].unique())
            for_map["Quantity"] = dataset.groupby(
                dataset["State"]).Quantity.sum().tolist()
            for_map['id'] = for_map['State'].apply(lambda x: state_id_map[x])

            # MAKING CHOROPLETH MAP
            fig = px.choropleth(for_map, locations='id',
                                geojson=indian_states,
                                color='Quantity',
                                hover_name='State',
                                scope='asia'
                                )
            fig.update_geos(fitbounds="locations", visible=True)
            st.header("Quantity of Sales by State")
            st.markdown('''
            Over here you have the **states** depicted in chorepleth/thematic map based on the **quantity of sales**.
            ''')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.empty()


# Market Basket Recommendation Function
with a_priori:
    try:
        # Setting making a list of default states
        defaultState = dataset['State'].unique()
        st.header("Market Basket Recommendation List")
        st.markdown('''
            Over here are the recommendation lists of item(s) from the selected **city** of the **state**.
        ''')

        # Make 2 columns
        customization_col, show_col = st.columns([1, 3])

        # First col
        with customization_col:
            # choose State
            stateName = st.selectbox(
                "Choose State:", options=sorted(defaultState), index=0)
            # Choose City Box
            chosen_cityName = st.selectbox(
                "Choose City:", options=dataset[dataset["State"] == stateName].City.unique().tolist(), index=0)

        # Second col
        with show_col:

            # Setting metrics for ARM (with defaultilizer)
            set_cityName = chosen_cityName
            minSPval = 0
            set_metric = ""
            set_min_threshold = 0

            # Deafult values setter - better safe than sorry
            try:
                if set_cityName == "":
                    set_cityName = dataset["State"].City.unique().tolist()[
                        0]
                if minSPval == 0:
                    minSPval = 0.1
                if set_metric == "":
                    set_metric = "lift"
                if set_min_threshold == 0:
                    set_min_threshold = 1
            except:
                default_cityName = dataset["State"].City.unique().tolist()[
                    0]
                minSPval = 0.1
                set_metric = "lift"
                set_min_threshold = 1

            # Making raw rules for ARM with frozensets
            basket = (dataset[dataset['City'] == set_cityName]
                      .groupby(['Order ID', 'Sub-Category'])['Quantity']
                      .sum().unstack().reset_index().fillna(0)
                      .set_index('Order ID')
                      )
            # Apply OHE into MHT df
            basket_encoded = basket.applymap(hot_encode)
            basket = basket_encoded

            # Building the model
            frq_items_basket = apriori(
                basket, min_support=0.1, use_colnames=True)

            # Collecting inferred rules in df
            rules_basket = association_rules(
                frq_items_basket, metric=set_metric, min_threshold=1)
            rules_basket = rules_basket.sort_values(
                ['confidence', 'lift'], ascending=[False, False])

            # Generated Rules (unjargonizer and user-friendly)
            # MAKE NEW DF WITH USER FRIENDLY TERMS
            report_df = pd.DataFrame()
            report_df["People who buy"] = rules_basket.antecedents.tolist()
            report_df["Will buy"] = rules_basket.consequents.tolist()
            report_df["Cross-selling metric"] = rules_basket.lift.tolist()

            # CALL AD-HOC UNJARGONIZER
            fSets_remove(report_df, "People who buy")
            fSets_remove(report_df, "Will buy")
            report_df = report_df.sort_values(
                ["Cross-selling metric"], ascending=False)
            report_df = report_df.style.hide_index()
            st.write(report_df)

        # Show more details about the ARM report (Market Basket Recommendation)
        with st.expander("More details"):

            # math columns - mCol
            mCol1, mCol2, mCol3 = st.columns(3)

            with mCol1:
                st.latex(r'''
                    \large Support = \dfrac{ freq(A,B) }{N}
                ''')
                st.markdown('''
                    *Frequency of item/itemsets over the entire dataset.*
                ''')
            with mCol2:
                st.latex(r'''
                    \large Confidence = \dfrac{ freq(A,B) }{ freq(A)}
                ''')
                st.markdown('''
                    *Frequency of itemsets given the antecedent item is purchased.* 
                    > Think of it as Bayes' theorem of Probability!
                ''')
            with mCol3:
                st.latex(r'''
                    \large Lift = \dfrac{ Supp (A,B) }{ Supp(A) \bullet Supp(B) }
                ''')
                st.markdown('''
                    *The correlation between both items/itemsets.* 
                    > As stated in the Market Recommendation List, it's the value of cross-selling opportunity!
                ''')

            # Get raw rules from rules_basket dataframe
            report = rules_basket

            # Calling AD-HOC Unjargonizer
            fSets_remove(report, "antecedents")
            fSets_remove(report, "consequents")

            # Dropping irrelevants
            report.drop('leverage', inplace=True, axis=1)
            report.drop('conviction', inplace=True, axis=1)
            st.write(report)
    except:
        st.empty()
