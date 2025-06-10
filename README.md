# Cash Reconciliation Matcher

A powerful Streamlit application for matching and reconciling cash transactions between sales and deposits. This tool helps identify potential matches between tender amounts and deposit amounts while considering various matching criteria and constraints.

## Features

- **Excel File Processing**: Upload and process Excel files containing sales and deposit data
- **Flexible Column Mapping**: Map your data columns to the required fields
- **Dynamic Filtering**: Apply multiple filters to focus on specific subsets of data
- **Configurable Matching Parameters**:
  - Amount tolerance for matching
  - Maximum combination size
  - Maximum date difference
  - Parent group size
  - Chain penalty options
- **Store-wise Processing**: Processes data store by store for better organization
- **Comprehensive Results**:
  - Detailed matching information
  - Summary statistics
  - Download options (CSV and Excel)

## Prerequisites

- Python 3.7+
- Required Python packages:
  - streamlit
  - pandas
  - numpy
  - openpyxl

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your Excel file containing the following data:
   - Tender Amount
   - Sales Date
   - Store ID
   - Deposit Date
   - Deposit Amount
   - Status

4. Configure the application:
   - Map your Excel columns to the required fields
   - Set up any necessary filters
   - Adjust matching parameters as needed

5. Click "Run Matching Process" to start the matching algorithm

6. View and download the results:
   - Review the matching results in the interactive table
   - Download results as CSV or Excel file

## Matching Parameters Explained

- **Amount Tolerance**: Maximum allowed difference between parent and child amounts
- **Maximum Combination Size**: Maximum number of children that can be matched to a parent
- **Maximum Date Difference**: Maximum allowed days between parent and child dates
- **Maximum Parent Group Size**: Maximum number of parents that can be grouped together
- **Chain Penalty**: Optional penalty for longer chains of matches

## Output

The application generates a comprehensive report including:
- Matched Parent ID
- Matched Parent Amount
- Effective Parent Amount
- Effective Parent-Child Difference
- All original data columns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
