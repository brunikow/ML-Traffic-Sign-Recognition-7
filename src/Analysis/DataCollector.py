import pandas as pd

"""
Class that can safe data in a table and save it in a csv file
"""
class DataCollector:
    """
    Initiates a DataCollector object. And saves some general information

    @param columns: Header of the table. The length of this list determines the number of columns.
    """
    def __init__(self, columns: list[str]) -> None:
        self.df = pd.DataFrame(columns=columns)
        self.num_of_columns = len(columns)
        self.counter = 0

    """
    Can be called to add a row to the table. Throws an error if row and table format missmatch.
    
    @param row: Row of data we want to save.
    """
    def collect(self, row: list) -> int:
        self.df.loc[len(self.df)] = row
        if(len(row) != self.num_of_columns):
            raise ValueError(f"Expected {self.num_of_columns} values, got {len(row)}")
        self.counter += 1
        return self.counter

    """
    Saves the table of this object.

    @param location: destination for the csv file.
    """
    def save_df(self, location: str) -> None:
        self.df.to_csv(location, index=False)
    