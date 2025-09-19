from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def _name(p: str | Path) -> str:
    """basename with extension (e.g., 'file.csv')."""
    try:
        return Path(p).name
    except Exception:
        return str(p)


def ts2min(ts: float, resolution: int) -> str:
    """Convert number of timestamps to 00:00 format

    Args:
        ts (int): timestamp
        resolution (int): e.g. 30000

    Returns:
        str: length of timestamps in 00:00 format
    """
    total_seconds = ts / resolution
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02}:{seconds:02}"


def ts2unix(time_origin, resolution, ts) -> datetime:
    """
    Convert a timestamp into a Unix timestamp
    based on the origin time and resolution.

    Args:
        time_origin: e.g. datetime.datetime(2024, 4, 16, 22, 7, 32, 403000)
        resolution: e.g. 30000
        ts: e.g. 37347215

    Returns:
        e.g. 2024-04-16 22:28:17.310167
    """
    base_time = datetime(
        time_origin.year,
        time_origin.month,
        time_origin.day,
        time_origin.hour,
        time_origin.minute,
        time_origin.second,
        time_origin.microsecond,
    )
    # division first prevents overflow
    microseconds = ts / resolution * 1000000
    return base_time + timedelta(microseconds=microseconds)


def to_16bit_binary(number: int) -> str:
    """convert number to 16-bit binary

    Args:
        number (int): e.g. 65535

    Returns:
        str: 1111111111111111
    """
    return format(number, "016b")


def make_bit_column(nev_digital_events_df, bit_number: int):
    """
    Make another column called "Bit{bit_number}"

    Args:
        nev_digital_events_df:
        InsertionReason 	TimeStamps 	UnparsedData 	UnparsedDataBin
    0 	1 	                149848003 	65535 	        1111111111111111
    1 	129 	            149848077 	45 	            0000000000101101
    2 	129 	            149848080 	39 	            0000000000100111
    3 	129 	            149848083 	33 	            0000000000100001

    Returns:
        df
        InsertionReason 	TimeStamps 	UnparsedData 	UnparsedDataBin    Bit{bit_number}
    0 	1 	                149848003 	65535 	        1111111111111111   1
    1 	129 	            149848077 	45 	            0000000000101101   0
    """
    df = nev_digital_events_df.copy()
    df[f"Bit{bit_number}"] = df["UnparsedDataBin"].apply(lambda x: int(x[bit_number]))
    return df


def fill_missing_data(nev_digital_events_df, bit_number: int):
    """
    Fill in the missing data points for UnparsedDataBin values so that every timestamp within the range is accounted for.

    Args:
        nev_digital_events_df (DataFrame): The DataFrame containing the data.
        bit_number (int): The nth bit starting from the left.

    Returns:
        filled_df (DataFrame): The DataFrame with missing data points filled.
    """
    # Create a new DataFrame with continuous timestamps
    df = make_bit_column(nev_digital_events_df, bit_number)
    df["TimeStamps"] = df["TimeStamps"].astype(int)
    min_timestamp = df["TimeStamps"].min()
    max_timestamp = df["TimeStamps"].max()
    all_timestamps = pd.DataFrame(
        {"TimeStamps": range(min_timestamp, max_timestamp + 1)}
    )

    # Merge the original DataFrame with the continuous timestamps DataFrame
    filled_df = all_timestamps.merge(df, on="TimeStamps", how="left")

    # Forward fill the missing values in the bit column
    filled_df[f"Bit{bit_number}"] = filled_df[f"Bit{bit_number}"].ffill().bfill()
    filled_df[f"Bit{bit_number}"] = filled_df[f"Bit{bit_number}"].astype(int)

    return filled_df


def fill_missing_serials_with_gap(data):
    """
    Fills in missing chunk serial numbers where the gap is greater than 1.
    The missing chunks are added with interpolated timestamps between the two existing ones.

    Parameters:
    -----------
    data : list of tuples
        Each tuple contains (timestamp, chunk_serial, UTCTimeStamp).

    Returns:
    --------
    list of tuples
        The list with the missing chunk serials filled in where appropriate.

    Example:
    --------
    >>> data = [(5412181557, 21428921, datetime(2024, 7, 26, 20, 30, 25, 509900)),
                (5412182558, 21428922, datetime(2024, 7, 26, 20, 30, 25, 543267)),
                (5412184559, 21428925, datetime(2024, 7, 26, 20, 30, 25, 609967))]
    >>> result = fill_missing_serials_with_gap(data)
    >>> for row in result:
    >>>     print(row)
    (5412181557, 21428921, datetime.datetime(2024, 7, 26, 20, 30, 25, 509900))
    (5412182558, 21428922, datetime.datetime(2024, 7, 26, 20, 30, 25, 543267))
    (5412183225, 21428923, datetime.datetime(2024, 7, 26, 20, 30, 25, 565500))
    (5412183891, 21428924, datetime.datetime(2024, 7, 26, 20, 30, 25, 587733))
    (5412184559, 21428925, datetime.datetime(2024, 7, 26, 20, 30, 25, 609967))
    """
    filled_data = []

    for i in range(len(data) - 1):
        # Append the current tuple to the result list
        filled_data.append(data[i])

        # Calculate the gap between consecutive chunk serial numbers
        current_serial = data[i][1]
        next_serial = data[i + 1][1]
        gap = next_serial - current_serial

        if gap > 1:
            # Calculate the time delta between the two timestamps
            time_delta = data[i + 1][2] - data[i][2]
            time_tsp_delta = data[i + 1][0] - data[i][0]

            # Populate missing serials
            for j in range(1, gap):
                new_serial = current_serial + j
                new_timestamp = data[i][2] + (time_delta / gap) * j
                new_timestp = data[i][0] + (time_tsp_delta // gap) * j
                filled_data.append((new_timestp, new_serial, new_timestamp))

    # Append the last tuple
    filled_data.append(data[-1])

    return filled_data
