"""
Library Features:

Name:          lib_dryes_downloader_os
Author(s):     Francesco Avanzi (francesco.avanzi@cimafoundation.org), Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20210929'
Version:       '1.0.0'
"""
#################################################################################
# Library
import os
#################################################################################

# -------------------------------------------------------------------------------------
# Method to make folder
def make_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
# -------------------------------------------------------------------------------------

