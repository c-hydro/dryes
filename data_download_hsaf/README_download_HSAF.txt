# Operative downloader (only H14)
## STEP1:
# DOWNLOAD H14 multiple layers daily dataset
Script: acquire_H14_v03.py
Conf: conf_h14.json

## STEP2:
# COMPUTE Root Zone Soil Moisture and store in final folder
Script: HSAF_regrid_RZSM_v3.py
Conf: conf_HSAF_RZSMtif.json


# LAST H14:
1. acquire_H14_v04_op.py
2. HSAF_regrid_RZSM_v4.py
