"""Perform a statistical analysis on the data."""


def statistical_analysis(df):
    """Perform counting of EC numbers"""
    value_counts = df['BRENDA'].value_counts().reset_index()
    value_counts.columns = ['BRENDA', 'Count']
 
    value_counts.to_csv('Output/brenda_ec_count.csv', index=False)