import streamlit as st
import pandas as pd
from itertools import combinations
import io
import openpyxl

def multi_parent_matching(
    parents_dict,  # e.g., {'p1': (1000.0, pd.Timestamp('2024-01-01')), ...}
    children_dict, # e.g., {'c1': (500.0, pd.Timestamp('2024-01-02')), ...}
    amount_tolerance=200,
    max_combo_size=3,
    use_chain_penalty=False,
    chain_penalty_weight=1.0,
    max_date_diff_days=None,  # new: max days between parent and child
    max_parent_group_size=3   # new: multi-parent matching support
):
    used_children = set()
    used_parents = set()
    matches = []

    parent_items = [(pid, data) for pid, data in parents_dict.items()]
    child_items = [(cid, data) for cid, data in children_dict.items()]

    for parent_group_size in range(1, max_parent_group_size + 1):
        for parent_combo in combinations(
            [p for p in parent_items if p[0] not in used_parents], parent_group_size
        ):
            parent_ids = [p[0] for p in parent_combo]
            parent_amt_sum = sum(p[1][0] for p in parent_combo)
            parent_dates = [p[1][1] for p in parent_combo]
            latest_parent_date = max(parent_dates)

            available_children = [
                (cid, data) for cid, data in child_items
                if cid not in used_children and data[1] <= latest_parent_date
            ]

            for r in range(1, min(max_combo_size, len(available_children)) + 1):
                for child_combo in combinations(available_children, r):
                    combo_ids = [x[0] for x in child_combo]
                    combo_amts = [x[1][0] for x in child_combo]
                    combo_dates = [x[1][1] for x in child_combo]

                    # Check if parent date >= all child dates
                    if any(min(parent_dates) < cd for cd in combo_dates):
                        continue

                    # Check max date difference if applicable
                    if max_date_diff_days is not None:
                        if any((latest_parent_date - cd).days > max_date_diff_days for cd in combo_dates):
                            continue

                    combo_amt = sum(combo_amts)
                    diff = abs(combo_amt - parent_amt_sum)

                    if diff <= amount_tolerance:
                        base_score = 1 / (1 + diff)
                        if use_chain_penalty:
                            penalty = 1 + chain_penalty_weight * (len(combo_ids) - 1)
                            score = base_score / penalty
                        else:
                            score = base_score

                        matches.append((
                            parent_ids,
                            combo_ids,
                            parent_amt_sum,
                            combo_amt,
                            diff,
                            latest_parent_date
                        ))

                        used_parents.update(parent_ids)
                        used_children.update(combo_ids)
                        break  # move to next parent combo after successful match

    return matches

st.set_page_config(page_title="Cash Reconciliation Matcher", layout="wide")

st.title("Cash Reconciliation Matcher")
st.write("Upload your Excel file and configure the matching parameters below.")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'csv'])

# Initialize session state for filters if not exists
if 'filters' not in st.session_state:
    st.session_state.filters = []

# Initialize session state for dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

# Initialize session state for column mappings
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {}

# Process file upload
if uploaded_file is not None and st.session_state.df is None:
    with st.spinner("Reading and processing file..."):
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

# Only show configuration if we have data
if st.session_state.df is not None:
    # Column mapping section
    st.header("Column Mapping")
    st.write("Verify the column mappings below. If any mapping is incorrect, select the correct column from the dropdown.")
    
    # Default column mappings
    default_mappings = {
        'Tender Amount': 'Tender Amount',
        'Sales Date': 'Sales Date',
        'Store ID': 'Store ID',
        'Deposit Date': 'Deposit Date',
        'Deposit Amount': 'Deposit Amount',
        'Status': 'Status'
    }
    
    # Create column mapping interface
    for required_col, default_col in default_mappings.items():
        available_columns = [col for col in st.session_state.df.columns]
        # Try to find exact match first
        if default_col in available_columns:
            default_value = default_col
        else:
            # If no exact match, use first available column
            default_value = available_columns[0] if available_columns else None
        
        st.session_state.column_mappings[required_col] = st.selectbox(
            f"Select column for {required_col}",
            options=available_columns,
            index=available_columns.index(default_value) if default_value in available_columns else 0,
            help=f"Default expected column name: {default_col}"
        )
    
    # Dynamic Filter Configuration
    st.header("Filter Configuration")
    
    # Add new filter button
    if st.button("Add New Filter"):
        st.session_state.filters.append({"column": None, "values": []})
    
    # Display existing filters
    for i, filter_config in enumerate(st.session_state.filters):
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            # Column selector
            filter_config["column"] = st.selectbox(
                f"Column {i+1}",
                options=st.session_state.df.columns,
                key=f"col_{i}"
            )
        
        with col2:
            # Value selector (multi-select)
            if filter_config["column"] is not None:
                unique_values = st.session_state.df[filter_config["column"]].unique()
                filter_config["values"] = st.multiselect(
                    f"Values for {filter_config['column']}",
                    options=unique_values,
                    key=f"vals_{i}"
                )
        
        with col3:
            # Remove filter button
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.filters.pop(i)
                st.rerun()
    
    # Matching parameters
    st.header("Matching Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount_tolerance = st.number_input(
            "Amount Tolerance",
            min_value=0,
            max_value=1000,
            value=200,
            help="Maximum allowed difference between parent and child amounts"
        )
        
        max_combo_size = st.number_input(
            "Maximum Combination Size",
            min_value=1,
            max_value=10,
            value=4,
            help="Maximum number of children that can be matched to a parent"
        )
    
    with col2:
        max_date_diff_days = st.number_input(
            "Maximum Date Difference (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Maximum allowed days between parent and child dates"
        )
        
        max_parent_group_size = st.number_input(
            "Maximum Parent Group Size",
            min_value=1,
            max_value=5,
            value=1,
            help="Maximum number of parents that can be grouped together"
        )
    
    with col3:
        use_chain_penalty = st.checkbox(
            "Use Chain Penalty",
            value=False,
            help="Apply penalty for longer chains of matches"
        )
        
        if use_chain_penalty:
            chain_penalty_weight = st.number_input(
                "Chain Penalty Weight",
                min_value=0.0,
                max_value=1000.0,
                value=1000.0,
                step=100.0
            )
        else:
            chain_penalty_weight = 0.0
    
    # Run matching process
    if st.button("Run Matching Process"):
        with st.spinner("Processing matches..."):
            print(amount_tolerance, max_combo_size, use_chain_penalty, chain_penalty_weight, max_date_diff_days, max_parent_group_size)
            
            # Rename columns based on mapping
            df = st.session_state.df.rename(columns={v: k for k, v in st.session_state.column_mappings.items()})
            
            # Apply filters
            filtered_df = df.copy()
            for filter_config in st.session_state.filters:
                if filter_config["column"] is not None and filter_config["values"]:
                    filtered_df = filtered_df[filtered_df[filter_config["column"]].isin(filter_config["values"])]
            
            match_groups_all = []
            
            # Get total number of stores for progress tracking
            total_stores = len(filtered_df['Store ID'].unique())
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a global mapping of child IDs to actual indices
            global_child_id_to_idx = {}
            current_child_idx = 0
            
            for idx, store_id in enumerate(filtered_df['Store ID'].unique()):
                # Update progress
                progress = (idx + 1) / total_stores
                progress_bar.progress(progress)
                status_text.text(f"Processing Store {store_id} ({idx + 1} of {total_stores})")
                
                valid_parents_df = filtered_df[(filtered_df["Deposit Amount"] > 0) & (filtered_df["Store ID"]==store_id)].copy()
                parents_dict = {
                    f"p_{idx}": (row["Deposit Amount"], row["Deposit Date"])
                    for idx, row in valid_parents_df.iterrows()
                }
                
                valid_child_df = filtered_df[(filtered_df["Tender Amount"] > 0) & (filtered_df["Store ID"]==store_id)].copy()
                children_dict = {
                    f"c_{idx}": (row["Tender Amount"], row["Sales Date"])
                    for idx, row in valid_child_df.iterrows()
                }
                
                if len(parents_dict)==0 or len(children_dict)==0:
                    continue
                    
                match_groups = multi_parent_matching(
                    parents_dict,
                    children_dict,
                    amount_tolerance=amount_tolerance,
                    max_combo_size=max_combo_size,
                    use_chain_penalty=use_chain_penalty,
                    chain_penalty_weight=chain_penalty_weight,
                    max_date_diff_days=max_date_diff_days,
                    max_parent_group_size=max_parent_group_size
                )
                match_groups_all += match_groups
            
            # Clear progress bar and status text
            progress_bar.empty()
            status_text.empty()
            
            # Process results
            # Initialize new columns
            results_df = df.copy()
            results_df["Matched Parent ID"] = None
            results_df["Matched Parent Amount"] = None
            results_df["Effective Parent Amount"] = None
            results_df["Effective Parent-Child Difference"] = None

            # Iterate over each matched group
            for parent_ids, child_ids, parent_amt, combo_amt, amt_diff, match_date in match_groups_all:
                running_amt = 0  # Running total of already assigned amounts
                # Add 2 to parent IDs for display
                parent_id_str = "+".join([f"p_{int(pid.split('_')[1]) + 2}" for pid in parent_ids])  # Add 2 to make it look better

                for i, child_id in enumerate(child_ids):
                    child_idx = int(child_id.split("_")[1])
                    tender_amt = results_df.at[child_idx, "Tender Amount"]

                    # Assign matched parent info
                    results_df.at[child_idx, "Matched Parent ID"] = parent_id_str
                    results_df.at[child_idx, "Matched Parent Amount"] = parent_amt

                    # Calculate effective parent amount
                    if i == len(child_ids) - 1:
                        eff_amt = parent_amt - running_amt  # Remainder for last child
                    else:
                        eff_amt = tender_amt
                        running_amt += tender_amt

                    # Store effective amount and difference
                    results_df.at[child_idx, "Effective Parent Amount"] = eff_amt
                    results_df.at[child_idx, "Effective Parent-Child Difference"] = eff_amt - tender_amt

            # Display results
            st.header("Matching Results")
            
            # Summary statistics
            total_matches = len(match_groups_all)
            total_parents = len(set([p for match in match_groups_all for p in match[0]]))
            total_children = len(set([c for match in match_groups_all for c in match[1]]))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", total_matches)
            with col2:
                st.metric("Total Parents Matched", total_parents)
            with col3:
                st.metric("Total Children Matched", total_children)
            
            # Display complete results
            st.dataframe(results_df, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Matching Results')
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    label="Download Results as Excel",
                    data=excel_data,
                    file_name="matching_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # CSV download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="matching_results.csv",
                    mime="text/csv"
                ) 