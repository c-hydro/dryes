import numpy as np
import xarray as xr
from typing import Optional

def cdi_v4(spi1:  np.ndarray,
           spi3:  Optional[np.ndarray] = None,
           sma:   Optional[np.ndarray] = None,
           fapar: Optional[np.ndarray] = None,
           cdi_p: Optional[np.ndarray] = None,
           count_sma_recovery:   Optional[np.ndarray] = None,
           count_fapar_recovery: Optional[np.ndarray] = None,
           domain: Optional[np.ndarray] = None
           ) -> tuple[np.ndarray]:
    """"
    Calculates the Combined Drought Indicator (CDI) version 4 based on the input parameters.
    For information on the CDI, see:
    https://drought.emergency.copernicus.eu/data/factsheets/factsheet_combinedDroughtIndicator_v4.pdf 

    Missing inputs are handled (see the function prepare_cdi_input):
        - by setting the corresponding layer to 255 for spi1, spi3, sma, fapar (no thresholds triggered).
        - by setting the corresponding layer to to 0 for cdi_p(assume previous CDI is always "no drought") and the counts.
    
    Parameters:
        spi1  (np.ndarray): Standardized Precipitation Index (SPI) for 1 month aggregation.
        spi3  (np.ndarray|None): Standardized Precipitation Index (SPI) for 3 months aggregation.
        sma   (np.ndarray|None): Soil Moisture Anomaly (SMA) values.
        fapar (np.ndarray|None): Fraction of Absorbed Photosynthetically Active Radiation (fAPAR) anomaly values.
        cdi_p (np.ndarray|None): Previous CDI values.
        count_sma_recovery   (np.ndarray|None): Count of SMA recovery periods.
        count_fapar_recovery (np.ndarray|None): Count of fAPAR recovery periods.
        domain (np.ndarray|None): Domain mask to apply to the CDI calculation.

    Returns:
        tuple: A tuple containing:
            - cdi (np.ndarray): The calculated CDI values.
            - count_sma_recovery (np.ndarray):   Updated count of SMA recovery periods (for the next timestep).
            - count_fapar_recovery (np.ndarray): Updated count of fAPAR recovery periods (for the next timestep).
            - cases (np.ndarray): The cases associated with the CDI calculation (for debugging purpose).
    """

    # Variables
    max_dek = 4 # maximum number of dekads
    th2 = -0.5 # sma threshold
    missingdata_value = 8 #missing data value
    cdi = np.zeros_like(cdi_p).astype('uint8') # starting CDI array
    cases = cdi.astype('int16') + 1000 # starting cases CDI array

    #zSPI (spim) with values of (0,1)
    a = np.logical_or.reduce((spi1 <= -2, spi3 <= -1))# from paper
    zspi = np.where(a == True, 1, 0)

    # Condition for case C and E
    b = np.logical_and.reduce((spi3 <= 0, spi3 > -1))
    c = np.logical_and.reduce((spi1 >-2 , spi1 <= -0.5))
    d = np.logical_and.reduce((b , c))
    spi13_cond = np.where(d == True, 1, 0)

    # New condition in case C
    e = np.logical_and.reduce((spi3 > 0 , spi1 > 0.5))
    spi13_cond_column_c = np.where(e == True, 1, 0)

    # SPI conditions
    spi13_cond0 = spi13_cond != 1
    spi13_cond1 = spi13_cond == 1
    zspi_0 = zspi != 1
    zspi_1 = zspi == 1
    # Soil Moisture conditions
    sma_gtm1 = sma > -1
    sma_lt025 = sma <= -0.25
    sma_ltth2 = sma <= th2
    sma_lt0 = sma <= 0
    sma_gt0 = sma > 0
    sma_gteth = sma >= th2 
    sma_gte025 = sma >= -0.25 
    sma_ltm1 = sma <= -1
    sma_gtth2 = sma > th2
    # fAPAR conditions
    fapar_gtm1 = fapar > -1
    fapar_gt0 = fapar > 0
    fapar_ltth = fapar <= th2
    fapar_gtth = fapar > th2
    fapar_lt0 = fapar <= 0
    fapar_ltm1 = fapar <= -1
    # Previuos CDI conditions
    cdi_p_0 = cdi_p == 0
    cdi_p_1 = cdi_p == 1
    cdi_p_2 = cdi_p == 2
    cdi_p_3 = cdi_p == 3
    cdi_p_4 = cdi_p == 4
    cdi_p_5 = cdi_p == 5
    cdi_p_6 = cdi_p == 6
    cdi_p_2_5 = cdi_p_2 | cdi_p_5
    cdi_p_3_6 = cdi_p_3 | cdi_p_6
    cdi_p_0_4 = cdi_p_0 | cdi_p_4
    cdi_p_1_4 = cdi_p_1 | cdi_p_4
    cdi_p_0_4_1 = cdi_p_0_4 | cdi_p_1
    cdi_p_1_2_5 = cdi_p_1 | cdi_p_2 | cdi_p_5
    cdi_p_0_4_1_2_5 = cdi_p_0_4_1 | cdi_p_2 | cdi_p_5
    cdi_p_1_2_3_5_6 = cdi_p_1 | cdi_p_2_5 | cdi_p_3_6

    ##  a Cases  
    # a1 case
    mask_case_a1 = zspi_0 & sma_gtm1 & fapar_gtm1 & cdi_p_0_4
    cdi[mask_case_a1] = 0
    cases[mask_case_a1] = 1010
    count_sma_recovery[mask_case_a1] = 0
    count_fapar_recovery[mask_case_a1] = 0
 
    # a2 case
    mask_case_a2 = zspi_0 & sma_gtm1 & fapar_gtm1 & cdi_p_1
    cdi[mask_case_a2] = 4
    cases[mask_case_a2] = 1024
    count_sma_recovery[mask_case_a2] = 0
    count_fapar_recovery[mask_case_a2] = 0

    # a3 case
    mask_case_a3 = zspi_0 & (sma_gtm1 & sma_ltth2) & fapar_gtm1 & cdi_p_2_5 & (count_sma_recovery <= max_dek)
    cdi[mask_case_a3] = 2
    cases[mask_case_a3] = 1032
    count_sma_recovery[mask_case_a3] = count_sma_recovery[mask_case_a3] + 1

    # a4 case
    mask_case_a4 = zspi_0 & fapar_gtm1 & cdi_p_2_5 & (sma_gtth2 & sma_lt0) & (count_sma_recovery <= max_dek)
    cdi[mask_case_a4] = 5
    cases[mask_case_a4] = 1045
    count_sma_recovery[mask_case_a4] = count_sma_recovery[mask_case_a4] + 1

    # Three complementary cases ***
    # a6 case
    mask_case_a6 = zspi_0 & (sma_gtm1 & sma_lt0) & fapar_gtm1 & cdi_p_2_5 & (count_sma_recovery > max_dek)
    cdi[mask_case_a6] = 4
    cases[mask_case_a6] = 1054
    count_sma_recovery[mask_case_a6] = 0

    # a5 case
    mask_case_a5 = zspi_0 & sma_gt0 & fapar_gtm1 & cdi_p_2_5 
    cdi[mask_case_a5] = 4
    cases[mask_case_a5] = 1064
    count_sma_recovery[mask_case_a5] = 0

    ## b Cases
    # b1 case
    mask_case_b11 = zspi_1 & (sma_gtm1 & sma_lt025) & fapar_gtm1 & cdi_p_0   
    cdi[mask_case_b11] = 1
    cases[mask_case_b11] = 1071
    count_sma_recovery[mask_case_b11] = 0
    count_fapar_recovery[mask_case_b11] = 0

    mask_case_b12 = zspi_1 & (sma > -0.25) & fapar_gtm1 & cdi_p_0 
    cdi[mask_case_b12] = 1
    cases[mask_case_b12] = 1081
    count_sma_recovery[mask_case_b12] = 0
    count_fapar_recovery[mask_case_b12] = 0

    mask_case_b13 = zspi_1 & sma_gtm1 & fapar_gtm1 & cdi_p_1_4
    cdi[mask_case_b13] = 1
    cases[mask_case_b13] = 1091
    count_sma_recovery[mask_case_b13] = 0
    count_fapar_recovery[mask_case_b13] = 0

    # b2 case
    mask_case_b2 = zspi_1 & (sma_gtm1 & sma_ltth2) & fapar_gtm1 & cdi_p_2_5 & (count_sma_recovery <= max_dek)
    cdi[mask_case_b2] = 2
    cases[mask_case_b2] = 1102
    count_sma_recovery[mask_case_b2] = count_sma_recovery[mask_case_b2] + 1
      
    # b3 case
    mask_case_b3 = zspi_1 & (sma_gtth2 & sma_lt0) & fapar_gtm1 & cdi_p_2_5 & (count_sma_recovery <= max_dek)
    cdi[mask_case_b3]= 5
    cases[mask_case_b3] = 1115
    count_sma_recovery[mask_case_b3] = count_sma_recovery[mask_case_b3] + 1

    # b4 case
    mask_case_b4 = zspi_1 & sma_gt0 & fapar_gtm1 & cdi_p_2_5 
    cdi[mask_case_b4]= 1
    cases[mask_case_b4] = 1121
    count_sma_recovery[mask_case_b4] = 0

    # b5 case
    mask_case_b5 = zspi_1 & (sma_gtm1 & sma_lt0 ) & fapar_gtm1 & cdi_p_2_5 & (count_sma_recovery > max_dek)
    cdi[mask_case_b5]= 1
    cases[mask_case_b5] = 1131
    count_sma_recovery[mask_case_b5] = 0

    ## c Cases
    # c1 case
    mask_case_c1 = zspi_0 & sma_ltm1 & fapar_gtm1 & cdi_p_0_4 & (spi13_cond_column_c == 1)
    cdi[mask_case_c1] = 0
    cases[mask_case_c1] = 1140
    count_sma_recovery[mask_case_c1] = 0
    count_fapar_recovery[mask_case_c1] = 0

    mask_case_c1a = zspi_0 & sma_ltm1 & fapar_gtm1 & cdi_p_0_4 & (spi13_cond_column_c != 1)
    cdi[mask_case_c1a] = 2
    cases[mask_case_c1a] = 1152
    count_sma_recovery[mask_case_c1a] = 0
    count_fapar_recovery[mask_case_c1a] = 0
    
    mask_case_c2 = zspi_0 & sma_ltm1 & fapar_gtm1 & cdi_p_1_2_5
    cdi[mask_case_c2] = 2
    cases[mask_case_c2] = 1162
    count_sma_recovery[mask_case_c2] = 0
    count_fapar_recovery[mask_case_c2] = 0

    ## d & e Cases
    #d1 case
    mask_case_d1 = zspi_0 & fapar_ltm1 & cdi_p_0_4 & sma_gtm1
    cdi[mask_case_d1] = 0
    cases[mask_case_d1] = 1170
    count_sma_recovery[mask_case_d1] = 0
    count_fapar_recovery[mask_case_d1] = 0

    #e0 case
    mask_case_e0 = zspi_0 & fapar_ltm1 & sma_ltm1 & cdi_p_0_4 & spi13_cond0
    cdi[mask_case_e0] = 3
    cases[mask_case_e0] = 1183
    count_sma_recovery[mask_case_e0] = 0
    count_fapar_recovery[mask_case_e0] = 0

    #e1 case
    mask_case_e1 = zspi_0 & fapar_ltm1 & sma_ltm1 & cdi_p_0_4 & spi13_cond1
    cdi[mask_case_e1] = 3
    cases[mask_case_e1] = 1193
    count_sma_recovery[mask_case_e1] = 0
    count_fapar_recovery[mask_case_e1] = 0

    #de2 case 
    mask_case_de2 = zspi_0 & fapar_ltm1 & cdi_p_1_2_3_5_6
    cdi[mask_case_de2] = 3
    cases[mask_case_de2] = 1203
    count_sma_recovery[mask_case_de2] = 0
    count_fapar_recovery[mask_case_de2] = 0
    
    #f1 case 
    mask_case_f1 = zspi_1 & sma_ltm1 & fapar_gtm1 & cdi_p_0_4_1_2_5
    cdi[mask_case_f1] = 2
    cases[mask_case_f1] = 1212
    count_sma_recovery[mask_case_f1] = 0
    count_fapar_recovery[mask_case_f1] = 0

    #case f + case c + case b + case a (special conditions of temporary)
    #a
    #w1 case
    mask_case_w1a = cdi_p_3_6 & (fapar_gtm1 & fapar_ltth) & (count_fapar_recovery <= max_dek) & zspi_0 & sma_gtm1
    cdi[mask_case_w1a]= 3
    cases[mask_case_w1a] = 1223
    count_fapar_recovery[mask_case_w1a] = count_fapar_recovery[mask_case_w1a] + 1

    #w2 case
    mask_case_w2a = cdi_p_3_6 & (fapar_gtth & fapar_lt0) & (count_fapar_recovery <= max_dek) & zspi_0 & sma_gtm1
    cdi[mask_case_w2a]= 6
    cases[mask_case_w2a] = 1236
    count_fapar_recovery[mask_case_w2a] = count_fapar_recovery[mask_case_w2a] + 1

    #w8  case
    mask_case_w8a = (fapar_gtm1 & fapar_lt0) & cdi_p_3_6 & sma_gtm1 & zspi_0 & (count_fapar_recovery > max_dek)  
    cdi[mask_case_w8a]= 4
    cases[mask_case_w8a] = 1244
    count_fapar_recovery[mask_case_w8a] = 0

    #w8 case
    mask_case_w7a = fapar_gt0 & cdi_p_3_6 & sma_gtm1 & zspi_0   
    cdi[mask_case_w7a]= 4
    cases[mask_case_w7a] = 1254
    count_fapar_recovery[mask_case_w7a] = 0

    #b
    #w1 case
    mask_case_w1b = (cdi_p_3_6 & (fapar_gtm1 & fapar_ltth) & (count_fapar_recovery <= max_dek)) & zspi_1 & sma_gtm1
    cdi[mask_case_w1b]= 3
    cases[mask_case_w1b] = 1263
    count_fapar_recovery[mask_case_w1b] = count_fapar_recovery[mask_case_w1b] +1

    #w2 case
    mask_case_w2b = (cdi_p_3_6 & (fapar_gtth & fapar_lt0) & (count_fapar_recovery <= max_dek)) & zspi_1 & sma_gtm1
    cdi[mask_case_w2b] = 6
    cases[mask_case_w2b] = 1276
    count_fapar_recovery[mask_case_w2b] = count_fapar_recovery[mask_case_w2b] +1

    #w3 case
    mask_case_w3b = (fapar_gtm1 & fapar_lt0) & cdi_p_3_6 & (count_fapar_recovery > max_dek) & sma_gtm1 & zspi_1   
    cdi[mask_case_w3b] = 1
    cases[mask_case_w3b] = 1281
    count_fapar_recovery[mask_case_w3b] = 0

    #w4 case
    mask_case_w4b = fapar_gt0 & cdi_p_3_6 & sma_gtm1 & zspi_1   
    cdi[mask_case_w4b]= 1
    cases[mask_case_w4b] = 1291
    count_fapar_recovery[mask_case_w4b] = 0

    #c
    #w1 case
    mask_case_w1c = cdi_p_3_6 & (fapar_gtm1 & fapar_ltth) & (count_fapar_recovery <= max_dek) & zspi_0 & sma_ltm1  
    cdi[mask_case_w1c]= 3
    cases[mask_case_w1c] = 1303
    count_fapar_recovery[mask_case_w1c] = count_fapar_recovery[mask_case_w1c] +1

    #w2 case
    mask_case_w2c = cdi_p_3_6 & (fapar_gtth & fapar_lt0) & (count_fapar_recovery <= max_dek) & zspi_0 & sma_ltm1 
    cdi[mask_case_w2c]= 6
    cases[mask_case_w2c] = 1316
    count_fapar_recovery[mask_case_w2c] = count_fapar_recovery[mask_case_w2c] + 1

    #w6 case
    mask_case_w6c = (fapar_gtm1 & fapar_lt0) & cdi_p_3_6 & sma_ltm1 & (count_fapar_recovery > max_dek) & zspi_0
    cdi[mask_case_w6c] = 2 
    cases[mask_case_w6c] = 1322
    count_fapar_recovery[mask_case_w6c] = 0    

    #w5 case   
    mask_case_w5c = fapar_gt0 & cdi_p_3_6 & sma_ltm1 & zspi_0
    cdi[mask_case_w5c] = 2 
    cases[mask_case_w5c] = 1332 
    count_fapar_recovery[mask_case_w5c] = 0   
    
    #f
    #w1 case
    mask_case_w1f = cdi_p_3_6 & (fapar_gtm1 & fapar_ltth) & (count_fapar_recovery <= max_dek) & zspi_1 & sma_ltm1 
    cdi[mask_case_w1f] = 3 
    cases[mask_case_w1f] = 1343 
    count_fapar_recovery[mask_case_w1f] = count_fapar_recovery[mask_case_w1f] + 1
    
    #w2 case
    mask_case_w2f = cdi_p_3_6 & (fapar_gtth & fapar_lt0) & (count_fapar_recovery <= max_dek) & zspi_1 & sma_ltm1 
    cdi[mask_case_w2f] = 6 
    cases[mask_case_w2f] = 1356
    count_fapar_recovery[mask_case_w2f] = count_fapar_recovery[mask_case_w2f]+ 1

    #w6 case
    mask_case_w6f = (fapar_gtm1  & fapar_lt0) & cdi_p_3_6 & sma_ltm1 & (count_fapar_recovery > max_dek) & zspi_1
    cdi[mask_case_w6f] = 2 
    cases[mask_case_w6f] = 1362
    count_fapar_recovery[mask_case_w6f] = 0 

    #w5 case
    mask_case_w5f = (fapar_gt0 & cdi_p_3_6 & sma_ltm1 & zspi_1)
    cdi[mask_case_w5f] = 2 
    cases[mask_case_w5f] = 1372
    count_fapar_recovery[mask_case_w5f] = 0  

    #cases g & h
    mask_case_gh = zspi_1 & fapar_ltm1 
    cdi[mask_case_gh]= 3
    cases[mask_case_gh] = 1383
    count_sma_recovery[mask_case_gh] = 0
    count_fapar_recovery[mask_case_gh] = 0

    ## mask the data based on the domain
    if domain is not None:
        cdi   = np.where(domain == 1, cdi, missingdata_value)
        cases = np.where(domain == 1, cases, 999)

    return  cdi, count_sma_recovery, count_fapar_recovery, cases

def cdi_norec(spi1:  np.ndarray,
              spi3:  Optional[np.ndarray] = None,
              sma:   Optional[np.ndarray] = None,
              fapar: Optional[np.ndarray] = None,
              cdi_p: Optional[np.ndarray] = None,
              domain: Optional[np.ndarray] = None
              ) -> tuple[np.ndarray]:
    """"
    Calculates the Combined Drought Indicator (CDI) based on the input parameters.
    This version of the CDI is based on the original CDI version 4 but does not include recovery periods:
    https://drought.emergency.copernicus.eu/data/factsheets/factsheet_combinedDroughtIndicator_v4.pdf

    Missing inputs are handled (see the function prepare_cdi_input):
        - by setting the corresponding layer to 255 for spi1, spi3, sma, fapar (no thresholds triggered).
        - by setting cdi_p to 0 (assume previous CDI is always "no drought").
    
    Parameters:
        spi1   (np.ndarray): Standardized Precipitation Index (SPI) for 1 month aggregation.
        spi3   (np.ndarray|None): Standardized Precipitation Index (SPI) for 3 months aggregation.
        sma    (np.ndarray|None): Soil Moisture Anomaly (SMA) values.
        fapar  (np.ndarray|None): Fraction of Absorbed Photosynthetically Active Radiation (fAPAR) anomaly values.
        cdi_p  (np.ndarray|None): Previous CDI values.
        domain (np.ndarray|None): Domain mask to apply to the CDI calculation.

    Returns:
        tuple: A tuple containing:
            - cdi (np.ndarray): The calculated CDI values.
            - cases (np.ndarray): The cases associated with the CDI calculation (for debugging purpose).
    """

    # Variables
    th2 = -0.5 # sma threshold
    missingdata_value = 8 #missing data value
    cdi = np.zeros_like(cdi_p).astype('uint8') # starting CDI array
    cases = cdi.astype('int16') + 1000 # starting cases CDI array

    #zSPI (spim) with values of (0,1)
    a = np.logical_or.reduce((spi1 <= -2, spi3 <= -1))# from paper
    zspi = np.where(a == True, 1, 0)

    # Condition for case C and E
    b = np.logical_and.reduce((spi3 <= 0, spi3 > -1))
    c = np.logical_and.reduce((spi1 > -2 , spi1 <= -0.5))
    d = np.logical_and.reduce((b , c))
    spi13_cond = np.where(d == True, 1, 0)

    # New condition in case C
    e = np.logical_and.reduce((spi3 > 0 , spi1 > 0.5))
    spi13_cond_column_c = np.where(e == True, 1, 0)

    # SPI conditions
    spi13_cond0 = spi13_cond != 1
    spi13_cond1 = spi13_cond == 1
    zspi_0 = zspi != 1
    zspi_1 = zspi == 1
    # Soil Moisture conditions
    sma_gtm1 = sma > -1
    sma_lt025 = sma <= -0.25
    sma_ltth2 = sma <= th2
    sma_lt0 = sma <= 0
    sma_gt0 = sma > 0
    sma_gteth = sma >= th2 
    sma_gte025 = sma >= -0.25 
    sma_ltm1 = sma <= -1
    sma_gtth2 = sma > th2
    # fAPAR conditions
    fapar_gtm1 = fapar > -1
    fapar_gt0 = fapar > 0
    fapar_ltth = fapar <= th2
    fapar_gtth = fapar > th2
    fapar_lt0 = fapar <= 0
    fapar_ltm1 = fapar <= -1
    # Previuos CDI conditions
    cdi_p_0 = cdi_p == 0
    cdi_p_1 = cdi_p == 1
    cdi_p_2 = cdi_p == 2
    cdi_p_3 = cdi_p == 3

    cdi_p_1_2 = cdi_p_1 | cdi_p_2
    cdi_p_0_1_2 = cdi_p_0 | cdi_p_1 | cdi_p_2
    cdi_p_1_2_3 = cdi_p_1 | cdi_p_2 | cdi_p_3

    ##  a Cases  
    # a1 case
    mask_case_a1 = zspi_0 & sma_gtm1 & fapar_gtm1 & cdi_p_0
    cdi[mask_case_a1] = 0
    cases[mask_case_a1] = 1010
 
    # a2 case
    mask_case_a2 = zspi_0 & sma_gtm1 & fapar_gtm1 & cdi_p_1
    cdi[mask_case_a2] = 0
    cases[mask_case_a2] = 1024

    # a3 case
    mask_case_a3 = zspi_0 & (sma_gtm1 & sma_ltth2) & fapar_gtm1 & cdi_p_2
    cdi[mask_case_a3] = 2
    cases[mask_case_a3] = 1032

    # a4 case
    mask_case_a4 = zspi_0 & (sma_gtth2 & sma_lt0) & fapar_gtm1 & cdi_p_2 
    cdi[mask_case_a4] = 1
    cases[mask_case_a4] = 1045

    # a5 case
    mask_case_a5 = zspi_0 & sma_gt0 & fapar_gtm1 & cdi_p_2
    cdi[mask_case_a5] = 0
    cases[mask_case_a5] = 1064

    ## b Cases
    # b1 case
    mask_case_b11 = zspi_1 & sma_gtm1 & fapar_gtm1 & cdi_p_0   
    cdi[mask_case_b11] = 1
    cases[mask_case_b11] = 1071

    mask_case_b13 = zspi_1 & sma_gtm1 & fapar_gtm1 & cdi_p_1
    cdi[mask_case_b13] = 1
    cases[mask_case_b13] = 1091

    # b2 case
    mask_case_b2 = zspi_1 & (sma_gtm1 & sma_ltth2) & fapar_gtm1 & cdi_p_2
    cdi[mask_case_b2] = 2
    cases[mask_case_b2] = 1102
      
    # b3 case
    mask_case_b3 = zspi_1 & (sma_gtth2 & sma_lt0) & fapar_gtm1 & cdi_p_2
    cdi[mask_case_b3]= 2
    cases[mask_case_b3] = 1115

    # b4 case
    mask_case_b4 = zspi_1 & sma_gt0 & fapar_gtm1 & cdi_p_2
    cdi[mask_case_b4]= 1
    cases[mask_case_b4] = 1121

    ## c Cases
    # c1 case
    mask_case_c1 = zspi_0 & sma_ltm1 & fapar_gtm1 & cdi_p_0 & (spi13_cond_column_c == 1)
    cdi[mask_case_c1] = 0
    cases[mask_case_c1] = 1140

    mask_case_c1a = zspi_0 & sma_ltm1 & fapar_gtm1 & cdi_p_0 & (spi13_cond_column_c != 1)
    cdi[mask_case_c1a] = 2
    cases[mask_case_c1a] = 1152
    
    mask_case_c2 = zspi_0 & sma_ltm1 & fapar_gtm1 & cdi_p_1_2
    cdi[mask_case_c2] = 2
    cases[mask_case_c2] = 1162

    ## d & e Cases
    #d1 case
    mask_case_d1 = zspi_0 & fapar_ltm1 & cdi_p_0 & sma_gtm1
    cdi[mask_case_d1] = 0
    cases[mask_case_d1] = 1170

    #e0 case
    mask_case_e0 = zspi_0 & fapar_ltm1 & sma_ltm1 & cdi_p_0 & spi13_cond0
    cdi[mask_case_e0] = 3
    cases[mask_case_e0] = 1183

    #e1 case
    mask_case_e1 = zspi_0 & fapar_ltm1 & sma_ltm1 & cdi_p_0 & spi13_cond1
    cdi[mask_case_e1] = 3
    cases[mask_case_e1] = 1193

    #de2 case 
    mask_case_de2 = zspi_0 & fapar_ltm1 & cdi_p_1_2_3
    cdi[mask_case_de2] = 3
    cases[mask_case_de2] = 1203
    
    #f1 case 
    mask_case_f1 = zspi_1 & sma_ltm1 & fapar_gtm1 & cdi_p_0_1_2
    cdi[mask_case_f1] = 2
    cases[mask_case_f1] = 1212

    #case f + case c + case b + case a (special conditions of temporary)
    #a
    #w1 case
    mask_case_w1a = cdi_p_3 & (fapar_gtm1 & fapar_ltth) & zspi_0 & sma_gtm1
    cdi[mask_case_w1a]= 3
    cases[mask_case_w1a] = 1223

    #w2 case
    mask_case_w2a = cdi_p_3 & (fapar_gtth & fapar_lt0) & zspi_0 & sma_gtm1
    cdi[mask_case_w2a]= 2

    #w8 case
    mask_case_w7a = cdi_p_3 & fapar_gt0 & sma_gtm1 & zspi_0   
    cdi[mask_case_w7a]= 1
    cases[mask_case_w7a] = 1254

    #b
    #w1 case
    mask_case_w1b = cdi_p_3 & (fapar_gtm1 & fapar_ltth) & zspi_1 & sma_gtm1
    cdi[mask_case_w1b]= 3
    cases[mask_case_w1b] = 1263

    #w2 case
    mask_case_w2b = cdi_p_3 & (fapar_gtth & fapar_lt0) & zspi_1 & sma_gtm1
    cdi[mask_case_w2b] = 2
    cases[mask_case_w2b] = 1276

    #w4 case
    mask_case_w4b = fapar_gt0 & cdi_p_3 & sma_gtm1 & zspi_1   
    cdi[mask_case_w4b]= 1
    cases[mask_case_w4b] = 1291

    #c
    #w1 case
    mask_case_w1c = cdi_p_3 & (fapar_gtm1 & fapar_ltth) & zspi_0 & sma_ltm1  
    cdi[mask_case_w1c]= 3
    cases[mask_case_w1c] = 1303

    #w2 case
    mask_case_w2c = cdi_p_3 & (fapar_gtth & fapar_lt0) & zspi_0 & sma_ltm1 
    cdi[mask_case_w2c]= 3
    cases[mask_case_w2c] = 1316   

    #w5 case   
    mask_case_w5c = fapar_gt0 & cdi_p_3 & sma_ltm1 & zspi_0
    cdi[mask_case_w5c] =  2
    cases[mask_case_w5c] = 1332   
    
    #f
    #w1 case
    mask_case_w1f = cdi_p_3 & (fapar_gtm1 & fapar_ltth) & zspi_1 & sma_ltm1 
    cdi[mask_case_w1f] = 3 
    cases[mask_case_w1f] = 1343 
    
    #w2 case
    mask_case_w2f = cdi_p_3 & (fapar_gtth & fapar_lt0) & zspi_1 & sma_ltm1 
    cdi[mask_case_w2f] = 3
    cases[mask_case_w2f] = 1356

    #w5 case
    mask_case_w5f = cdi_p_3 & fapar_gt0 & sma_ltm1 & zspi_1
    cdi[mask_case_w5f] = 2 
    cases[mask_case_w5f] = 1372 

    #cases g & h
    mask_case_gh = zspi_1 & fapar_ltm1 
    cdi[mask_case_gh]= 3
    cases[mask_case_gh] = 1383

    ## mask the data based on the domain
    if domain is not None:
        cdi   = np.where(domain == 1, cdi, missingdata_value)
        cases = np.where(domain == 1, cases, 999)

    return cdi, cases

def prepare_cdi_inputs(spi1:  np.ndarray,
                       spi3:  np.ndarray|None = None,
                       sma:   np.ndarray|None = None,
                       fapar: np.ndarray|None = None,
                       cdi_p: np.ndarray|None = None,
                       count_sma_recovery:   np.ndarray|None = None,
                       count_fapar_recovery: np.ndarray|None = None,
                       cascade_nans: bool = True
                       ) -> tuple[np.ndarray]: 
    """
    Prepare the inputs for the CDI calculation by masking and setting nodata values.

    For the SPI1, SPI3, SMA and fAPAR inputs:
    1. If the input is missing or None, all values are set to 255.
    2. If the input is not None, the missing (nodata) values are set to 255.
    3. The missign inputs are cascaded if cascade_nans is True:
        meaning that where SPI1 is missing, SPI3 is also set to 255, and so on.
        the cascading order is SPI1 -> SPI3 -> SMA -> fAPAR.

    For the previous CDI and the counts:
    1. If the input is missing or None, all values are set to 0.
    2. If the input is not None, the missing (nodata) values are set to 0.

    Parameters:
        spi1  (np.ndarray): Standardized Precipitation Index (SPI) for 1 month aggregation.
        spi3  (np.ndarray|None): Standardized Precipitation Index (SPI) for 3 months aggregation.
        sma   (np.ndarray|None): Soil Moisture Anomaly (SMA) values.
        fapar (np.ndarray|None): Fraction of Absorbed Photosynthetically Active Radiation (fAPAR) anomaly values.
        cdi_p (np.ndarray|None): Previous CDI values.
        count_sma_recovery   (np.ndarray|None): Count of SMA recovery periods.
        count_fapar_recovery (np.ndarray|None): Count of fAPAR recovery periods.
        cascade_nans (bool): If True, cascade the nodata values from SPI1 to SPI3, SMA, and fAPAR.
    Returns:
        tuple: A tuple containing:
            - spi1  (np.ndarray): The SPI1 values with nodata values set to 255.
            - spi3  (np.ndarray): The SPI3 values with nodata values set to 255.
            - sma   (np.ndarray): The SMA values with nodata values set to 255.
            - fapar (np.ndarray): The fAPAR values with nodata values set to 255.
            - cdi_p (np.ndarray): The previous CDI values with nodata values set to 0.
            - count_sma_recovery   (np.ndarray): The count of SMA recovery periods with nodata values set to 0.
            - count_fapar_recovery (np.ndarray): The count of fAPAR recovery periods with nodata values set to 0.
    """

    # set the nodata values to 255 (above all thresholds)
    nan_value = 255

    # start with SPI1, which is the only mandatory input
    spi1 = np.where(np.isnan(spi1),  nan_value, spi1)

    # continue with SPI3, SMA and fAPAR
    # for these:
    # 1. check if input is None, if it is, set all to nan_value = 255
    # 2. if not, set the nodata values to nan_value
    # 3. cascade nans if cascade_nans = True (i.e. set SPI3 to nan_value where SPI1 is nan_value, set SMA to nan_value where SPI3 is nan_value, etc.)

    if spi3 is None:
        spi3 = np.full_like(spi1, nan_value)
    else:
        spi3 = np.where(np.isnan(spi3), nan_value, spi3)
        if cascade_nans:
            spi3 = np.where(np.isclose(spi1, nan_value), nan_value, spi3)

    if sma is None:
        sma = np.full_like(spi1, nan_value)
    else:
        sma = np.where(np.isnan(sma), nan_value, sma)
        if cascade_nans:
            sma = np.where(np.isclose(spi3, nan_value), nan_value, sma)
    
    if fapar is None:
        fapar = np.full_like(spi1, nan_value)
    else:
        fapar = np.where(np.isnan(fapar), nan_value, fapar)
        if cascade_nans:
            fapar = np.where(np.isclose(sma, nan_value), nan_value, fapar)

    # Do the same for cdi_p but no cascade here
    cdi_nanval = 8
    if cdi_p is None:
        cdi_p = np.zeros_like(spi1)
    else:
        cdi_p = np.where(np.isclose(cdi_p, cdi_nanval, equal_nan=True), 0, cdi_p)

    # the counts shouldn't have nans
    if count_fapar_recovery is None:
        count_fapar_recovery = np.zeros_like(spi1)

    if count_sma_recovery is None:
        count_sma_recovery = np.zeros_like(spi1)

    return spi1, spi3, sma, fapar, cdi_p, count_sma_recovery, count_fapar_recovery