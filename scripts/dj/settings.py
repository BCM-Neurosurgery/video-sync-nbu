DATABASE_NAME = "emu24_stitch"
DATALAKE_PATH = "/mnt/datalake/data/emu/"
STITCHED_PATH = "/mnt/stitched/EMU-18112"
LOGGING_PATH = "/mnt/lake-database/stitched-logs/datajoint_computed_table.log"
DJ_CONFIG_SAFEMODE = True
DJ_CONFIG_STORES = {
    "Ext_Chunk": {
        "protocol": "file",
        "location": DATALAKE_PATH,
        "stage": DATALAKE_PATH,
    },
    "Ext_Stitch": {
        "protocol": "file",
        "location": STITCHED_PATH,
        "stage": STITCHED_PATH,
    },
}
