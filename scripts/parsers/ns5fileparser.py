import pandas as pd
from brpylib import NsxFile
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
from scripts.utility.utils import (
    ts2min,
    ts2unix,
)


class Nsx:
    def __init__(self, path) -> None:
        self.path = path
        self.nsxObj = NsxFile(path)
        self.nsxDict = vars(self.nsxObj)
        self.nsxData = self.nsxObj.getdata()
        self.nsxObj.close()
        self.init_vars()

    def init_vars(self):
        self.basic_header = self.nsxDict["basic_header"]
        self.extended_headers = self.nsxDict["extended_headers"]
        self.timestampResolution = self.basic_header["TimeStampResolution"]
        self.sampleResolution = self.basic_header["SampleResolution"]
        self.timeOrigin = self.basic_header["TimeOrigin"]
        self.extended_headers_df = pd.DataFrame.from_records(
            self.get_extended_headers()
        )
        self.data = self.nsxData
        self.memmapData = self.data["data"][0]
        # TODO: the data header might have multiple timestamps
        self.timeStamp = self.data["data_headers"][0]["Timestamp"]
        self.numDataPoints = self.data["data_headers"][0]["NumDataPoints"]
        self.recording_duration_s = self.data["data_headers"][0]["data_time_s"]
        self.recording_duration_readable = ts2min(self.recording_duration_s, 1)

    def get_start_timestamp(self):
        return self.timeStamp

    def get_timeOrigin(self):
        return self.timeOrigin

    def get_duration_readable(self):
        return self.recording_duration_readable

    def get_basic_header(self):
        return self.basic_header

    def get_data(self):
        return self.data

    def get_extended_headers(self) -> List[dict]:
        return self.extended_headers

    def get_extended_headers_df(self) -> pd.DataFrame:
        return self.extended_headers_df

    def get_sample_resolution(self):
        return self.sampleResolution

    def get_num_data_points(self):
        return self.numDataPoints

    def get_recording_duration_s(self):
        return self.recording_duration_s

    def get_channel_array(self, channel: str):
        """
        Args:
            channel: e.g. "RoomMic2"
        """
        row_index = self.extended_headers_df[
            self.extended_headers_df["ElectrodeLabel"] == channel
        ].index.item()
        return self.memmapData[row_index]

    def preview_channel_rows(
        self,
        channel: str,
        head: int = 5,
        tail: int = 5,
        include_utc: bool = True,
    ) -> str:
        """
        Efficiently preview the first and last few rows for a channel without
        materializing the entire DataFrame.

        Builds tiny slices from the underlying memmap and formats them as text.

        Args:
            channel: Name of the channel (matches Extended Header ElectrodeLabel).
            head: Number of rows to show from the start.
            tail: Number of rows to show from the end.
            include_utc: Include the UTCTimeStamp column in the preview.

        Returns:
            A formatted string containing head and tail rows.

        Notes:
            - If head + tail exceeds available samples, tail is reduced to avoid duplication.
            - This does not allocate arrays for the full recording; only small slices are created.
        """
        # Retrieve channel memmap (zero-copy view)
        channel_data = self.get_channel_array(channel)
        n = int(len(channel_data))

        if n == 0:
            return f"Channel '{channel}': no samples"

        head_n = max(0, min(head, n))
        tail_n = max(0, min(tail, n - head_n))  # avoid overlapping rows

        ts0 = int(self.timeStamp)

        # Build head slice
        head_amp = channel_data[:head_n]
        head_ts = np.arange(ts0, ts0 + head_n)
        if include_utc:
            head_utc = [
                ts2unix(self.timeOrigin, self.timestampResolution, ts - self.timeStamp)
                for ts in head_ts
            ]
        # Build tail slice
        tail_amp = channel_data[n - tail_n : n] if tail_n > 0 else []
        tail_ts_start = ts0 + (n - tail_n)
        tail_ts = np.arange(tail_ts_start, ts0 + n) if tail_n > 0 else np.array([])
        if include_utc and tail_n > 0:
            tail_utc = [
                ts2unix(self.timeOrigin, self.timestampResolution, ts - self.timeStamp)
                for ts in tail_ts
            ]

        # Assemble small DataFrames for nice column alignment
        cols = {"TimeStamp": head_ts, "Amplitude": head_amp}
        if include_utc:
            cols["UTCTimeStamp"] = head_utc
        head_df = pd.DataFrame(cols)

        if tail_n > 0:
            cols_tail = {"TimeStamp": tail_ts, "Amplitude": tail_amp}
            if include_utc:
                cols_tail["UTCTimeStamp"] = tail_utc
            tail_df = pd.DataFrame(cols_tail)
        else:
            tail_df = pd.DataFrame(columns=head_df.columns)

        # Format output text
        lines = []
        lines.append(
            f"Channel: {channel} | samples: {n} | start_ts: {ts0} | sample_res: {self.sampleResolution} | ts_res: {self.timestampResolution}"
        )
        lines.append("")
        lines.append(f"First {head_n} row(s):")
        lines.append(head_df.to_string(index=False))
        if tail_n > 0:
            lines.append("\n…\n")
            lines.append(f"Last {tail_n} row(s):")
            lines.append(tail_df.to_string(index=False))

        return "\n".join(lines)
