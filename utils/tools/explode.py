# Explode a cloud into one cloud per label


import pdal # For laz reading
import sys

try:
    filename=sys.argv[1]
except IndexError:
    print("Put filename in second argument")
# The filter for PDAL, in json format

for i in range(15):
    json = """
    [
        "%s",
        {
            "type":"filters.range",
            "limits":"Classification[%s:%s]"
        },
        {
            "type":"writers.las",
            "filename":"%s.laz",
            "forward":"all"
        }
    ]
    """ % (filename, str(i), str(i), str(i))
    
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    print(str(i) + "/15 finish !")
