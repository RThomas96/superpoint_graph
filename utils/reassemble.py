# Reassemble an exploded cloud

import pdal # For laz reading
import sys

try:
    filename=sys.argv[1]
except IndexError:
    print("No filename specified, default.laz used")
    filename = "default.laz"

for i in range(15):
    json = """
    [
        "%s.laz",
        {
            "type":"filters.assign",
            "value":["Classification = %s"]
        },
        {
            "type":"writers.las",
            "filename":"%s.laz",
            "forward":"all"
        }
    ]
    """ % (str(i), str(i), str(i))
    
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()

json = """
[
    "0.laz",
    "1.laz",
    "2.laz",
    "3.laz",
    "4.laz",
    "5.laz",
    "6.laz",
    "7.laz",
    "8.laz",
    "9.laz",
    "10.laz",
    "11.laz",
    "12.laz",
    "13.laz",
    "14.laz",
    {
        "type": "filters.merge"
    },
    {
        "type":"writers.las",
        "filename":"%s",
        "forward":"all"
    }
]
""" % (filename)

pipeline = pdal.Pipeline(json)
count = pipeline.execute()
